import logging, pathlib, time, signal, numpy as np
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from pythonrecordingclient import session
import MCloud, MCloudPacketRCV, MCloudPacketSND #type: ignore


BASE_DIR = pathlib.Path(__file__).resolve().parent


prc_logger = logging.getLogger('prc')
prc_logger.setLevel(logging.DEBUG)
prc_logger.addHandler(
    logging.FileHandler(BASE_DIR/'router.log', mode='a+')
)

HOST = "i13srv53.ira.uka.de"
W_PORT = 60019
C_PORT = 4443

W_NAME = "LID_Router_Worker"
C_NAME = "LID_Router_Client"
W_IN_FINGERPRINT = 'xx'
C_IN_FINGERPRINT = ('yy', 'zz')
IN_TYPE = 'audio'
W_OUT_FINGERPRINT = 'xx'
C_OUT_FINGERPRINT = ('yy', 'zz')
OUT_TYPE = 'text'

C_STREAM_IDS = ('1234', '5678')

SMAPLE_RATE = 16000


clients: list[session.Session] = []

def stop_clients():
    while clients:
        cl = clients.pop()
        print(f"Stopping client {cl.input_fingerprint}")
        cl.stop()
        print(f"Stopped client {cl.input_fingerprint}")


switch_count = 1
switch_cap = 10
switch_index = 0


def stop_gracefully(signal, frame):
    print("Stopping clients")
    stop_clients()
    print("Stopped clients")
    exit(0)

signal.signal(signal.SIGINT, stop_gracefully)



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
    for in_fp, out_fp, stream_id in zip(C_IN_FINGERPRINT, C_OUT_FINGERPRINT, C_STREAM_IDS):
        cl = session.Session(
            name=C_NAME,
            input_fingerprint=in_fp,
            input_types=[IN_TYPE],
            output_fingerprint=[out_fp],
        )
        cl.start(HOST, C_PORT)

        clients.append(cl)

        print(f"Successfully created client {in_fp}")

    for cl in clients:
        while not cl.ready:
            time.sleep(0.1)
        print(f"Client {cl.input_fingerprint} is ready")



@session.on_receive
def data_return_callback(cl, data: ET.Element):
    prc_logger.debug(f" <<< {ET.tostring(data, encoding='unicode', method='xml')}")
    text = unquote(data.find('text').text)
    start = data.get('start')
    start_int = int( data.get('startoffset') )
    stop = data.get('stop')
    stop_int = int( data.get('stopoffset') )
    creator = data.get('creator')
    fingerprint = data.get('fingerprint')
    print(f"Text: {text}\nStart: {start}\nStop: {stop}\nCreator: {creator}\nFingerprint: {fingerprint}")
    m_cloud_w.send_packet_result_async(start_int, stop_int, [text], 1)





def data_callback(i, sampleA):
    print('DEBUG', 'i:', type(i), i)
    print('DEBUG', 'sampleA:', type(sampleA), len(sampleA) )
    sample = np.array(sampleA, dtype=np.int16)
    print('DEBUG', 'sample:', type(sample), sample.shape, sample.min(), sample.max() )

    cl = get_client()

    cl.send_audio(sample.data)

    print('DEBUG', f"packet sent to client {cl.input_fingerprint}")
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

    
    


m_cloud_w = MCloud.MCloudWrap(W_NAME.encode('utf-8'), 1) # Mode 1: Worker

m_cloud_w.add_service(W_NAME.encode('utf-8'), 'lid'.encode('utf-8'), W_IN_FINGERPRINT.encode('utf-8'), IN_TYPE.encode('utf-8'), W_OUT_FINGERPRINT.encode('utf-8'), OUT_TYPE.encode('utf-8'), ''.encode('utf-8'))
m_cloud_w.set_callback("init", init_callback)
m_cloud_w.set_data_callback("worker")
m_cloud_w.set_callback("finalize", processing_finalize_callback)
m_cloud_w.set_callback("error", processing_error_callback)
m_cloud_w.set_callback("break", processing_break_callback)

m_cloud_w.connect(HOST.encode('utf-8'), W_PORT)



while True:

    while m_cloud_w.wait_for_client('123'.encode('utf-8')) == 1:
        time.sleep(1)

    proceed = True
    while proceed:
        packet = MCloudPacketRCV.MCloudPacketRCV(m_cloud_w)

        with open('packets.log', 'ba+') as f:
            f.write(packet.xml_string)
            f.write(b'\n\n')

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

            for cl in clients:
                cl.send_flush()

            m_cloud_w.send_flush()

            print("WORKER INFO received flush message ==> waiting for packages.")
            MCloud.mcloudpacketdenit(packet)
            continue
        elif packet.packet_type() == 4:  # MCloudDone
            m_cloud_w.wait_for_finish(1, "processing")

            stop_clients()
            
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            print("WOKRER INFO received DONE message ==> waiting for clients.")
            proceed = False
        elif packet.packet_type() == 5:  # MCloudError
            # In case of a error or reset message, the processing is stopped immediately by
            # calling m_cloud_wBreak followed by exiting the thread.
            m_cloud_w.wait_for_finish(1, "processing")

            #TODO handle error for clients
            stop_clients()
            
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            print("WORKER INFO received ERROR message >>> waiting for clients.")
            proceed = False
        elif packet.packet_type() == 6:  # MCloudReset
            m_cloud_w.stop_processing("processing")

            #TODO handle reset for clients
            stop_clients()
            raise NotImplemented

            print("CLIENT INFO received RESET message >>> waiting for clients.")
            MCloud.mcloudpacketdenit(packet)
            proceed = False
        else:
            print("CLIENT ERROR unknown packet type {!s}".format(packet.packet_type()))

            stop_clients()

            MCloud.mcloudpacketdenit(packet)
            proceed = False