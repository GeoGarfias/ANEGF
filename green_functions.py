#======================================================================================================================
# M O D U L E S
import os
import sys
import h5py
import obspy   
import argparse
import datetime
import numpy as np
import pandas as pd

from obspy import read, read_inventory

from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime


from obspy.signal.rotate import rotate_ne_rt
from obspy.geodetics.base import gps2dist_azimuth

from scipy.signal import butter, freqz
from scipy.fft import rfft, irfft, next_fast_len

#======================================================================================================================

def basics(tr):
    "Basic stuff for a seismic trace"
    #for tr in st:
    tr.detrend('demean')
    tr.detrend('simple')
    tr.taper(max_percentage=0.05)
    tr.filter("bandpass", freqmin=0.05, freqmax=1)
    tr.resample(sampling_rate=10,no_filter=True)
    tr.detrend('demean')
    tr.detrend('simple')
    return tr

def len_tr(stime,etime,st):
    '''
    Make traces the same length
    '''
    st.trim(starttime=stime,endtime=etime,pad=True,fill_value=0)
    return(st)

def moving_ave(A, N: int):
    """
    Alternative function for moving average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the full!! window length to smooth
    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    """
    # defines an array with N extra samples at either side
    temp = np.zeros(len(A) + 2 * N)
    # set the central portion of the array to A
    temp[N:-N] = A
    # leading samples: equal to first sample of actual array
    temp[0:N] = temp[N]
    # trailing samples: Equal to last sample of actual array
    temp[-N:] = temp[-N - 1]
    # convolve with a boxcar and normalize, and use only central portion of the result
    # with length equal to the original array, discarding the added leading and trailing samples
    B = np.convolve(temp, np.ones(N) / N, mode="same")[N:-N]
    return B

def filter(fs,n,lowcut,highcut,size):
    '''
    '''
    fc = np.array([lowcut,highcut])
    w = 2*fc/fs # Normalize the frequency
    b,a = butter(n,w,btype='bandpass')
    w,h = freqz(b,a,worN=size)

    return h

def calculate_gf(source: obspy.Trace,receiver: obspy.Trace, metho:str):
    '''
    Function to do crosscorelation, deconvolutions and coherency.
    :param source:                  (obspy.Trace) Source station data. Three channels. Downsampled to 10 Hz
    :param receiver:                (obspy.Trace) Receiver station data. Three channels. Downsampled to 10 Hz
    :param metho:                    (str) methood to use. Options: cross, deco, cohe, all

    :return trcross:                (obspy.Trace) Cross-correlation between source and receiver.
    :return trdeco:                 (obspy.Trace) Deconvolution between source and receiver.
    :return trcohe:                 (obspy.Trace) Coherency between source and receiver.
    '''
    #This works for both source and receiver
    nfft = next_fast_len(source.stats.npts*10) 

    sf = rfft(source.data,nfft)
    rf = rfft(receiver.data,nfft)

    #Calculate crosscorrelation in the frequency domain
    rcrossf = rf * np.conj(sf)
    trcross = np.real(np.fft.ifftshift(irfft(rcrossf, axis=0))) # Do not touch crosscorrelation, its all good the way it is.

    pts = 20
    if metho == 'cross':
        return trcross
    elif metho == 'deco':
        #Calculate deconvolution in the frequency domain
        temps = moving_ave(np.absolute(sf),pts)

        decof = rcrossf/ (temps**2)

        bfilter = filter(source.stats.sampling_rate,4,0.05,1,len(decof)) 

        deco = decof * bfilter
        trdeco = np.real(np.fft.ifftshift(irfft(deco, axis=0)))

        return trdeco
    elif metho == 'cohe':
        #Calculate coherency in the frequency domain
        temps = moving_ave(np.absolute(sf),pts)
        tempr = moving_ave(np.absolute(rf),pts) 

        cohef = rcrossf / (temps * tempr)

        bfilter = filter(source.stats.sampling_rate,4,0.05,1,len(cohef)) 

        cohe = cohef * bfilter
        trcohe = np.real(np.fft.ifftshift(irfft(cohe, axis=0)))

        return trcohe
    elif metho == 'all':
        temps = moving_ave(np.absolute(sf),pts)
        tempr = moving_ave(np.absolute(rf),pts) 

        decof = rcrossf/ (temps**2)
        cohef = rcrossf / (temps * tempr)

        bfilter = filter(source.stats.sampling_rate,4,0.05,1,len(decof)) 

        deco = decof * bfilter
        cohe = cohef * bfilter

        trdeco = np.real(np.fft.ifftshift(irfft(deco, axis=0)))
        trcohe = np.real(np.fft.ifftshift(irfft(cohe, axis=0)))

        return trcross, trdeco, trcohe


def bazimuth(source:str, receiver:str, s_net: str, r_net: str, path: str):
    '''
    Function to calculate the back-azimuth between two stations

    :param source:                  (str) Name of the source station.
    :param receiver:                (str) Name of the receiver station.
    :param s_net:                   (str) Network of the source station.
    :param r_net:                   (str) Network of the receiver station.

    :return rs_baz:                 (float) Back-azimuth from the receiver to the source.
    :return sr_baz:                 (float) Back-azimuth from the source to the receiver.
    '''
    salsa_inv_path = np.loadtxt(fname=path + 'stationxml_paths.txt', dtype='str')[0]
    geonet_inv_path = np.loadtxt(fname=path + 'stationxml_paths.txt', dtype='str')[1]
    inv_s = read_inventory(salsa_inv_path)
    inv_g = read_inventory(geonet_inv_path)

    if s_net or r_net == 'ZX':
        for i in range(len(inv_s[0])):
            if source == inv_s[0][i].code:
                s_lon = inv_s[0][i].longitude
                s_lat = inv_s[0][i].latitude
            if receiver == inv_s[0][i].code:
                r_lon = inv_s[0][i].longitude
                r_lat = inv_s[0][i].latitude
    if s_net or r_net == 'NZ':
        for i in range(len(inv_g[0])):
            if source == inv_g[0][i].code:
                s_lon = inv_g[0][i].longitude
                s_lat = inv_g[0][i].latitude
            if receiver == inv_g[0][i].code:
                r_lon = inv_g[0][i].longitude
                r_lat = inv_g[0][i].latitude

    d, rs_baz, sr_baz = gps2dist_azimuth(s_lat,s_lon,r_lat,r_lon)
   
    return rs_baz, sr_baz

def rotate(source: obspy.Trace, receiver: obspy.Trace, path: str):
    '''
    Function to rotate the data to the radial and transverse components
    :param source:                  (obspy.Stream) Source station data. 
    :param receiver:                (obspy.Stream) Receiver station data.

    :return source:                 (obspy.Stream) Rotated source station data. Three channels - Vertical, Radial and Transverse.
    :return receiver:               (obspy.Stream) Rotated receiver station data. Three channels - Vertical, Radial and Transverse.
    '''
    s_net = source[0].stats.network
    r_net = receiver[0].stats.network

    s_name = source[0].stats.station
    r_name = receiver[0].stats.station
    
    r_baz, s_baz = bazimuth(s_name,r_name,s_net,r_net, path)

    if s_name == 'WHSZ':
        loca = source[0].stats.location
        if loca == '10':
            source.rotate(method='NE->RT',back_azimuth=s_baz)
        elif loca == '11':
            s_radial, s_tranverse = rotate_ne_rt(source.select(component='1')[0].data,source.select(component='2')[0].data,s_baz)
            s_rad = source.select(component='1')[0].stats
            s_tran = source.select(component='2')[0].stats
            s_rad.channel = 'HHR'
            s_tran.channel = 'HHT'
            str_rad = Trace(data=s_radial,header=s_rad)
            str_tran = Trace(data=s_tranverse,header=s_tran)
            source = Stream(traces=[str_rad,str_tran,source.select(component='Z')[0]])
    elif r_name == 'WHSZ':
        loca = receiver[0].stats.location
        if loca == '10':
            receiver.rotate(method='NE->RT',back_azimuth=s_baz)
        elif loca == '11':
            r_radial, r_tranverse = rotate_ne_rt(receiver.select(component='1')[0].data,receiver.select(component='2')[0].data,r_baz)
            r_rad = receiver.select(component='1')[0].stats
            r_tran = receiver.select(component='2')[0].stats
            r_rad.channel = 'HHR'
            r_tran.channel = 'HHT'
            rtr_rad = Trace(data=r_radial,header=r_rad)
            rtr_tran = Trace(data=r_tranverse,header=r_tran)
            receiver = Stream(traces=[rtr_rad,rtr_tran,receiver.select(component='Z')[0]])
   
    if s_net == 'ZX':
        s_radial, s_tranverse = rotate_ne_rt(source.select(component='1')[0].data,source.select(component='2')[0].data,s_baz)
        s_rad = source.select(component='1')[0].stats
        s_tran = source.select(component='2')[0].stats
        s_rad.channel = 'HHR'
        s_tran.channel = 'HHT'
        str_rad = Trace(data=s_radial,header=s_rad)
        str_tran = Trace(data=s_tranverse,header=s_tran)
        source = Stream(traces=[str_rad,str_tran,source.select(component='Z')[0]])
    elif s_net == 'NZ':
        source.rotate(method='NE->RT',back_azimuth=s_baz)

    if r_net == 'ZX':
        r_radial, r_tranverse = rotate_ne_rt(receiver.select(component='1')[0].data,receiver.select(component='2')[0].data,r_baz)
        r_rad = receiver.select(component='1')[0].stats
        r_tran = receiver.select(component='2')[0].stats
        r_rad.channel = 'HHR'
        r_tran.channel = 'HHT'
        rtr_rad = Trace(data=r_radial,header=r_rad)
        rtr_tran = Trace(data=r_tranverse,header=r_tran)
        receiver = Stream(traces=[rtr_rad,rtr_tran,receiver.select(component='Z')[0]])
    elif r_net == 'NZ':
        receiver.rotate(method='NE->RT',back_azimuth=r_baz)

    return source, receiver

def cut_data(gf):
    '''
    Function that saves the crosscorrelation, deconvolution and coherency data.
    It cuts the data to have positive and negative time lags of 1000s

    '''

    time = np.linspace(-(len(gf)/2),(len(gf)/2),len(gf))*0.1
    time_range = (-600,600)
    time_mask = np.logical_and(time >= time_range[0], time <= time_range[1])
    
    ngf = gf[time_mask]
    ngf = ngf.astype(np.float32)

    return ngf

def check_data(source:obspy.Stream,receiver: obspy.Stream, path: str):
    '''
    Function that check if the data is good for crosscorrelations, etc.
    Checks if there is a minimum of 12 hours of data in the the source and receiver traces. If not, source =0 and receiver = 0
    Checks if there are 3 channels. If there are less than 3 channel looks for the vertical component. If not vertical component then source =0 and receiver = 0
    '''
    
    acceptable_time = 3600 # One hour in seconds

    if len(source) != 0 and len(receiver) != 0:
        for tr in source:
            if tr.stats.endtime - tr.stats.starttime == acceptable_time:
                tr = tr
            else:
                source.remove(tr)

        for tr in receiver:
            if tr.stats.endtime - tr.stats.starttime == acceptable_time:
                tr = tr
            else:
                receiver.remove(tr)
    
        if len(source) == 3 and len(receiver) == 3:
            source, receiver = rotate(source,receiver, path)
        elif len(source) <= 2 or len(receiver) <= 2:
            try:
                source = source.select(component='Z')[0]
                receiver = receiver.select(component='Z')[0]
            except:
                source = 0
                receiver = 0
    else:
        source = 0
        receiver = 0
        
    return source, receiver

def process(source: obspy.Stream,receiver: obspy.Stream,day: int,metho: str,path: str):
    '''
    Function that performs cross-correlation, deconvolution or coherency.
    :param source:                  (obspy.Stream) Source station data. Three channels. 
    :param receiver:                (obspy.Stream) Receiver station data. Three channels.
    :param day:                    (int) Day of analysis. In juliand days.
    :param metho:                    (str) methood to use. Options: cross, deco, cohe, all
    '''   

    starttime = UTCDateTime(datetime.datetime(2022,1,1,0,30) + datetime.timedelta(day-1))
    endtime = UTCDateTime(datetime.datetime(2022,1,1,23,30) + datetime.timedelta(day-1)) #Only using 23 hours to have exact 1hr long windows.

    print('Real code starts')

    green = {'Crosscorrelation':{},'Deconvolution':{},'Coherency':{}} # This is the variable that will be saved in the h5 file

    hr = 3600
    t0 = starttime

    while t0 < endtime:
        t1 = t0 + hr
        _source = source.slice(t0,t1)
        _receiver = receiver.slice(t0,t1)

        _source,_receiver = check_data(_source,_receiver,path)

        if _source != 0 and _receiver != 0:
            _source = basics(_source)
            _receiver = basics(_receiver)

            _source = Stream(_source)
            _receiver = Stream(_receiver)

            for s in _source:
                for r in _receiver:

                    #Channel combination
                    chan = str(s.stats.station) + '.' + str(s.stats.channel) + '-' + str(r.stats.station) + '.' + str(r.stats.channel)
            
                    # Calculate the standard deviation of the source and receiver trace
                    source_sd = np.std(s.data)*10
                    receiver_sd = np.std(r.data)*10
                    # Max spikes in the data
                    source_max = np.max(s.data)
                    receiver_max = np.max(r.data)
                
                    if source_max < source_sd and receiver_max < receiver_sd:
                        if metho == 'all':
                            corr, deco, cohe = calculate_gf(s,r,metho)
                            ncorr = cut_data(corr)
                            ndeco = cut_data(deco)
                            ncohe = cut_data(cohe)
                        elif metho == 'cross':
                            corr =  calculate_gf(s,r,metho)
                            ncorr = cut_data(corr)                    
                        elif metho == 'deco':
                            deco = calculate_gf(s,r,metho)
                            ndeco = cut_data(deco)                          
                        elif metho == 'cohe':
                            cohe = calculate_gf(s,r,metho)
                            ncohe = cut_data(cohe)
                            
                    else:
                        print('Skip correlating this section with a source and/or receiver data higher than 10x its standard deviation')
                        continue
                    # Save the data in the green variable
                    if metho == 'all':
                        if chan not in green['Crosscorrelation']:
                            green['Crosscorrelation'][chan] = [np.array([])]
                        green['Crosscorrelation'][chan].append(ncorr)  # append the numpy array
        
                        if chan not in green['Deconvolution']:
                           green['Deconvolution'][chan] = [np.array([])]
                        green['Deconvolution'][chan].append(ndeco)  # append the numpy array
                        
                        if chan not in green['Coherency']:
                            green['Coherency'][chan] = [np.array([])]
                        green['Coherency'][chan].append(ncohe)  # append the numpy array
                    elif metho == 'cross':
                        if chan not in green['Crosscorrelation']:
                            green['Crosscorrelation'][chan] = [np.array([])]
                        green['Crosscorrelation'][chan].append(ncorr)
                    elif metho == 'deco':
                        if chan not in green['Deconvolution']:
                            green['Deconvolution'][chan] = [np.array([])]
                        green['Deconvolution'][chan].append(ndeco)
                    elif metho == 'cohe':
                        if chan not in green['Coherency']:
                            green['Coherency'][chan] = [np.array([])]
                        green['Coherency'][chan].append(ncohe)  
        else:
            print('Not enough data')
        t0 = t0 + hr

    return green

def save_files(my_dict,sname,rname,path,year):
    '''
    Function to save the files in h5 format
    :param my_dict:                 (dict) Dictionary with the green functions
    :param sname:                   (str) Name of stations source.
    :param rname:                   (str) Name of stations receiver.
    :param path:                    (str) Path to the Running_files folder.
    '''

    #HERE at the moment 1/12/2023
    saving_path = np.loadtxt(fname=path + 'saving_paths.txt', dtype='str')
    key1 = (list(my_dict.keys()))

    if sname == 'WHSZ1':
        file_name = str(saving_path) + '/' + str(year) + '/' + sname[0:4] + '/' + sname[0:4] + '-' + rname + '/' + sname + '-' + rname + '_' + str(key1[0]) + '-' + str(key1[-1]) + '.h5'
    elif sname == 'WHSZ2':
        file_name = str(saving_path) + '/' + str(year) + '/' + sname[0:4] + '/' + sname[0:4] + '-' + rname + '/' + sname + '-' + rname + '_' + str(key1[0]) + '-' + str(key1[-1]) + '.h5'
    else:
        folder_name = str(saving_path) + '/' + str(year) + '/' + sname + '/' + sname + '-' + rname + '/' 
        os.makedirs(folder_name,exist_ok=True)
        file_name = folder_name + sname + '-' + rname + '_' + str(key1[0]) + '-' + str(key1[-1]) + '.h5'

    with h5py.File(file_name,'w') as hdf:
        for index_day, day in enumerate(my_dict):
            d = hdf.create_group(str(day))
            for index_method, method in enumerate(my_dict[day]):
                m = d.create_group(method)
                for index_channel, channel in enumerate(my_dict[day][method]):
                    my_dict[day][method][channel].pop(0)
                    m.create_dataset(channel,data=my_dict[day][method][channel],compression='gzip', compression_opts=9)
        
    message = 'Files saved'
    return message    
                
        
def read_data(sname: str,rname: str,day: str,year: int, path:str):
    '''
    Function to read mseed data
    
    :param sname:                   (str) Name of stations source.
    :param rname:                   (str) Name of stations receiver. 
    :param day:                     (int) Day of analysis. In juliand days.
    :param year:                    (int) Year of analysis.
    :param path:                    (str) Path to the Running_files folder.


    :output source:                 (obspy.Stream) Source station data. Three channels. Downsampled to 10 Hz
    :output receiver:               (obspy.Stream) Receiver station data. Three channels. Downsampled to 10 Hz

    if not using Geonet or SALSA data change network codes for sources and receivers.
    '''

    
    receiver_stats = np.loadtxt(path + 'Geonet_names.txt', dtype='str')
    source_stats = np.loadtxt(fname=path + 'SALSA_names.txt', dtype='str')

    source_path = np.loadtxt(fname=path + 'waveforms_paths.txt', dtype='str')[0]
    receiver_path = np.loadtxt(fname=path + 'waveforms_paths.txt', dtype='str')[1]

    source = Stream()
    receiver = Stream()

    #print('Reading waveforms')

    if sname == 'WHSZ1':
        try:
            source = read(receiver_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(sname[0:4]) + '.NZ/*.10-HH*')
        except:
            source = 0
    elif sname == 'WHSZ2':
        try:
            source = read(receiver_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(sname[0:4]) + '.NZ/*.11-HH*')
        except:
            source = 0

    if sname in source_stats:
        try:
            source = read(source_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(sname) + '.ZX/*HH*')
        except:
            source = 0
    elif sname in receiver_stats:
        try:
            source = read(receiver_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(sname) + '.NZ/*HH*')
        except:
            source = 0

    if rname == 'WHSZ1':
        try:
            receiver = read(receiver_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(rname[0:4]) + '.NZ/*.10-HH*')
        except:
            receiver = 0
    elif rname == 'WHSZ2':
        try:
            receiver = read(receiver_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(rname[0:4]) + '.NZ/*.11-HH*')
        except:
            receiver = 0

    if rname in source_stats:
        try:
            receiver = read(source_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(rname) + '.ZX/*HH*')
        except:
            receiver = 0
    elif rname in receiver_stats:
        try:
            receiver = read(receiver_path + str(year) + '/' + str(year) + '.' + str(day).zfill(3) + '/' + str(rname) + '.NZ/*HH*')
        except:
            receiver = 0

    if source != 0 and receiver !=0:
        source.merge(method=1, fill_value=0)
        receiver.merge(method=1, fill_value=0)
    else:
        pass

    #print('Finished reading and preprocessing waveforms')

    return source, receiver

def main_function(sname: str,rname: str,sday: int,eday: int,year: int,metho: str, path: str,start:datetime):
    '''
    This is the main function 
    '''
    print('Code starts')
    my_dict = {}
    julian_days = range(int(sday),int(eday)+1)
    for day in (julian_days):
        print(day)
        source, receiver = read_data(sname,rname,day,year,path)
        if source != 0 and receiver != 0:
            green = process(source,receiver,day,metho,path)
            my_dict[day] = green
        else:
            print('No data')
            continue
    
    m = save_files(my_dict,sname,rname,path,year)
    print(m)
    end = datetime.datetime.now()
    total_time = end - start
    print('Cross-correlation between source: ', str(sname), 'and receiver: ', str(rname), 'took: ', total_time)


def get_variables_to_run(id: int ,path: str): 
    '''
    This function gets the variables source_name, receiver_name, start_day, end_day, year, method from the id number of the job
    Input: 
    :param id:                                  ID number of the job
    :param start:                               The starttime of the code

    :output source_name:                        Name of the source station
    :output receiver_name:                      Name of the receiver station
    :output start_day:                          Start day of the analysis
    :output end_day:                            End day of the analysis
    :output year:                               Year of the analysis
    :output method:                             Method to use for the analysis: 'all', 'cross', 'deco', 'cohe'
    '''

    r_file = pd.read_csv(path + 'GF_runningfile.csv')
    # Choosing the row of the id number
    row = r_file[r_file['id'] == id]
    # Getting the variables
    source_name = row['source'].values[0]
    receiver_name = row['receiver'].values[0]
    start_day = row['start'].values[0]
    end_day = row['end'].values[0]
    year = row['year'].values[0]
    method = row['method'].values[0]

    print('Station source: ', source_name)
    print('Station receiver: ', receiver_name)
    print('Year: ', year)
    print('Days from ', start_day, 'to ', end_day)
    return source_name, receiver_name, start_day, end_day, year, method

def proc_main(start: datetime,path: str):
    '''
    This function is for reading the command line options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-id','--id_number',dest='id_number', help='ID number of the job', type=int,
                        required=True,action='store')
    args = parser.parse_args()
    id = args.id_number
    
    source_name, receiver_name, start_day, end_day, year, method = get_variables_to_run(id,path)

    main_function(source_name,receiver_name,start_day,end_day,year,method,path,start)
    

if __name__ == '__main__':
    '''
    This is the start of the code. To make it run you need to add the master path here.
    The master path is for the Running_files folder. The Running_files folder has to have the next files:
    - Geonet_names.txt    - Receivers names in a txt file
    - SALSA_names.txt     - Source names in a txt file
    - Geonet_inventory.xml  - Stationxml file for the receivers
    - SALSA_inventory.xml   - Stationxml file for the sources
    - csv_runningfile.csv   - A csv file with the source name, receiver namer, start time, end time, year, method and id number (from 0 to n)

    This a code to calculate cross-correlation, deconvolution and coherency for ambient seismic noise.
    The code uses parser for command-line options
    
    If want to debug the code, change debug = True & proc = False
    If you want to just run the code, proc = True & debug = False

    To run the code in proc mode, run it in the terminal like: 
    python green_functions.py -id 1
    where id is in the combination between source and receiver that is on the csv_runningfile.csv file.

    The modules needed to run are on the top of this code
    '''

    debug = False
    proc = True
    start = datetime.datetime.now()

    # MASTER PATH
    path = '/Users/home/juarezilma/PhD/Green/Running_files/'
    #path = '/nesi/project/vuw03918/Running_files/'

    if proc:
        proc_main(start,path)
    elif debug:
        #id = 6
        #source_name, receiver_name, start_day, end_day, year, method = get_variables_to_run(id,path)
        source_name = 'HAUP'
        receiver_name = 'MQZ'
        year = 2022
        start_day = 1
        end_day = 30
        method = 'cross'
        main_function(source_name,receiver_name,start_day,end_day,year,method,path,start)

        
