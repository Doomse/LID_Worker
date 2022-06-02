import time, threading, datetime, sys, numpy as np
import MCloud, MCloudPacketRCV, MCloudPacketSND #type: ignore


HOST = "i13srv53.ira.uka.de".encode('utf-8')
W_PORT = 60019
C_PORT = 4443

W_NAME = "LID_Router_Worker".encode('utf-8')
C_NAME = "LID_Router_Client".encode('utf-8')
W_IN_FINGERPRINT = 'xx'.encode('utf-8')
C_IN_FINGERPRINT = ('yy'.encode('utf-8'), 'zz'.encode('utf-8'))
IN_TYPE = 'audio'.encode('utf-8')
W_OUT_FINGERPRINT = 'xx'.encode('utf-8')
C_OUT_FINGERPRINT = ('yy'.encode('utf-8'), 'zz'.encode('utf-8'))
OUT_TYPE = 'unseg-text'.encode('utf-8')

C_STREAM_IDS = ('1234'.encode('utf-8'), '5678'.encode('utf-8'))

SMAPLE_RATE = 16000


clients = []

switch_count = 1
switch_cap = 10
switch_index = 0


#TODO replace this
def pass_on_client_responses(client):
    running = True
    while running:
        packet = MCloudPacketRCV.MCloudPacketRCV(client)

        print(f"receive worker packet {packet}", flush=True)

        p_type = packet.packet_type()
        if p_type == 3:
            client.process_data_async(packet, m_cloud_w)
        else:
            running = False

            



def processing_finalize_callback():
    print("INFO in processing finalize callback")
    #TODO teardown maybe here
    #print('DEBUG', 'translations', type(m_cloud_w.translations), len(m_cloud_w.translations))
    print("INFO finalize after print translation")

def processing_error_callback():
    print("INFO In processing error callback")
    #TODO cleanup

def processing_break_callback():
    print("INFO in processing break callback")
    #TODO cleanup

def init_callback():
    print("INFO in processing init callback ")
    running = True
    for in_fp, out_fp, stream_id in zip(C_IN_FINGERPRINT, C_OUT_FINGERPRINT, C_STREAM_IDS):
        m_cloud_c = MCloud.MCloudWrap(C_NAME, 2) # Mode 2: Client
        m_cloud_c.add_flow_description('en'.encode('utf-8'), 'lid'.encode('utf-8'), 'desc'.encode('utf-8'))
        m_cloud_c.set_callback("init", init_callback)
        m_cloud_c.set_data_callback("client")
        m_cloud_c.set_callback("finalize", processing_finalize_callback)
        m_cloud_c.set_callback("error", processing_error_callback)
        m_cloud_c.set_callback("break", processing_break_callback)

        m_cloud_c.connect(HOST, C_PORT)

        m_cloud_c.announce_output_stream(IN_TYPE, in_fp, stream_id, ''.encode('utf-8'))
        m_cloud_c.request_input_stream(OUT_TYPE, out_fp, stream_id)

        cl_thread = threading.Thread(target=pass_on_client_responses, args=(m_cloud_c, ) )

        clients.append( (m_cloud_c, in_fp, out_fp, stream_id, cl_thread) )

        print(f"Successfully created client {in_fp}")



def data_callback(i, sampleA):
    print('DEBUG', 'i:', type(i), i)
    print('DEBUG', 'sampleA:', type(sampleA), len(sampleA) )
    sample = np.asarray(sampleA)
    print('DEBUG', 'sample:', type(sample), sample.shape, sample.min(), sample.max() )

    now = datetime.datetime.now()
    end_str = now.strftime('%d/%m/%y-%H:%M:%S')
    td = datetime.timedelta(microseconds=i*1000000//16000 ) # num_frames/frames_per_second * microseconds_per_second
    start_str = (now-td).strftime('%d/%m/%y-%H:%M:%S')
    mcloud_cl, in_fp, out_fp, stream_id, cl_thread = get_client()
    snd_packet = MCloudPacketSND.MCloudPacketSND(mcloud_cl, 
        start_str.encode('utf-8'), 
        end_str.encode('utf-8'), 
        in_fp, 
        ''.encode('utf-8'), 
        b''.join([ s.to_bytes(2, sys.byteorder, signed=True) for s in sampleA]), 
        i, 
        0, 
    )
    mcloud_cl.send_packet(snd_packet) # Async sending doesn't work
    print('DEBUG', f"packet sent to client {in_fp}")
#[251, 260, 290, 298, 306, 315, 284, 294, 291, 371] byteorder='little'


def get_client():
    global switch_count
    global switch_index
    global switch_cap
    global clients
    if switch_count % switch_cap == 0:
        switch_count = 0
        switch_index += 1
        if switch_index == len(clients):
            switch_index = 0
    switch_count += 1
    return clients[switch_index]

    
    


m_cloud_w = MCloud.MCloudWrap(W_NAME, 1) # Mode 1: Worker

m_cloud_w.add_service(W_NAME, 'lid'.encode('utf-8'), W_IN_FINGERPRINT, IN_TYPE, W_OUT_FINGERPRINT, OUT_TYPE, ''.encode('utf-8'))
m_cloud_w.set_callback("init", init_callback)
m_cloud_w.set_data_callback("worker")
m_cloud_w.set_callback("finalize", processing_finalize_callback)
m_cloud_w.set_callback("error", processing_error_callback)
m_cloud_w.set_callback("break", processing_break_callback)

m_cloud_w.connect(HOST, W_PORT)



while True:

    while m_cloud_w.wait_for_client('123'.encode('utf-8')) == 1:
        time.sleep(1)

    proceed = True
    while proceed:
        packet = MCloudPacketRCV.MCloudPacketRCV(m_cloud_w)

        p_type = packet.packet_type()
        if packet.packet_type() == 3:
            m_cloud_w.process_data_async(packet, data_callback)
        elif packet.packet_type() == 7:  # MCloudFlush
            """
            a flush message has been received -> wait (block) until all pending packages
            from the processing queue has been processed -> finalizeCallback will
            be called-> flush message will be passed to subsequent components
            """
            m_cloud_w.wait_for_finish(0, "processing")

            for cl, _, _, _, _ in clients:
                cl.send_flush()

            print("WORKER INFO received flush message ==> waiting for packages.")
            MCloud.mcloudpacketdenit(packet)
            continue
        elif packet.packet_type() == 4:  # MCloudDone
            m_cloud_w.wait_for_finish(1, "processing")

            for cl, _, _, _, _ in clients:
                cl.send_done()

            
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            print("WOKRER INFO received DONE message ==> waiting for clients.")
            proceed = False
        elif packet.packet_type() == 5:  # MCloudError
            # In case of a error or reset message, the processing is stopped immediately by
            # calling m_cloud_wBreak followed by exiting the thread.
            m_cloud_w.wait_for_finish(1, "processing")

            #TODO handle error for clients
            for cl, _, _, _, _ in clients:
                cl.send_done()

            
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            print("WORKER INFO received ERROR message >>> waiting for clients.")
            proceed = False
        elif packet.packet_type() == 6:  # MCloudReset
            m_cloud_w.stop_processing("processing")

            #TODO handle reset for clients
            for cl, _, _, _, _ in clients:
                cl.send_done()

            print("CLIENT INFO received RESET message >>> waiting for clients.")
            MCloud.mcloudpacketdenit(packet)
            proceed = False
        else:
            print("CLIENT ERROR unknown packet type {!s}".format(packet.packet_type()))

            for cl, _, _, _, _ in clients:
                cl.send_done()

            MCloud.mcloudpacketdenit(packet)
            proceed = False