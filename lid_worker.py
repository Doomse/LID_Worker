import logging, pathlib, time, signal, numpy as np
import threading, time, queue
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from pythonrecordingclient import session
import MCloud, MCloudPacketRCV #type: ignore

from blackbox_runner import ModelRunner
#from stitching_runner import ModelRunner


BASE_DIR = pathlib.Path(__file__).resolve().parent


prc_logger = logging.getLogger('prc')
prc_logger.setLevel(logging.DEBUG)
prc_logger.addHandler(
    logging.FileHandler(BASE_DIR/'router.log', mode='a+')
)

lid_logger = logging.getLogger('lid')
lid_logger.setLevel(logging.DEBUG)
lid_logger.addHandler(
    logging.FileHandler(BASE_DIR/'lid.log', mode='a+')
)

HOST = "i13srv53.ira.uka.de"
W_PORT = 60019
C_PORT = 4443

W_NAME = "LID_Router_Worker"
C_NAME = "LID_Router_Client"
W_IN_FINGERPRINT = 'xx'
C_IN_FINGERPRINT = ('de', 'en')
IN_TYPE = 'audio'
W_OUT_FINGERPRINT = 'xx'
C_OUT_FINGERPRINT = ('de', 'en')
OUT_TYPE = 'text'

SAMPLE_RATE = 16000
MAPPING = {'de': 'de', 'en':'en', 'es':'de', 'fr':'de', 'it':'de', 'ru':'de'} #TODO check mappings
MODELFILE = BASE_DIR/'models'/'de,en__3__0.001__0.2____2022_08_16__10_45_17'

count = 0 #TODO move to router

class ClientRouter:

    def __init__(self, sr, model, mapping) -> None: #TODO Maybe pass all constants

        self.clients: dict[str, session.Session] = {}

        self.mapping = mapping
        self.runner = ModelRunner(MODELFILE, SAMPLE_RATE)

        self.send_queue = queue.Queue()

        self.processing = True

        for in_fp, out_fp in zip(C_IN_FINGERPRINT, C_OUT_FINGERPRINT):
            cl = session.Session(
                name=C_NAME,
                input_fingerprint=in_fp,
                input_types=[IN_TYPE],
                output_fingerprint=[out_fp],
            )
            cl.start(HOST, C_PORT)

            self.clients[in_fp] = cl

            print(f"Successfully created client {in_fp}")

        for cl in self.clients.values():
            while not cl.ready:
                time.sleep(0.1)
            print(f"Client {cl.input_fingerprint} is ready")
        
        self.current_language = ''

        self.send_thread = threading.Thread(target=self.send_audio)
        self.send_thread.start()


    def close(self):

        for cl in self.clients.values():
            print(f"Stopping client {cl.input_fingerprint}")
            cl.stop()
            print(f"Stopped client {cl.input_fingerprint}")

        #TODO stop threads
        self.processing = False
        self.send_thread.join()

        global count
        count = 0

    
    def process_audio(self, sample: np.ndarray):
        result = self.runner.run_model(sample)

        if not result is None:

            buffer: np.ndarray
            lang: str
            buffer, lang = result

            lang = self.mapping[lang]

            self.send_queue.put( (buffer, lang) )

            lid_logger.debug(f"packet sent to client {cl.input_fingerprint}")


    def send_audio(self): # Threaded method
        while self.processing or not self.send_queue.empty():

            buffer: np.ndarray
            lang:str
            try:
                buffer, lang = self.send_queue.get(block=True, timeout=5)
            except queue.Empty:
                continue

            if not self.current_language:
                self.current_language = lang
            if self.current_language != lang:
                self.clients[self.current_language].send_flush()
                self.current_language = lang
            
            self.clients[lang].send_audio(buffer.data)

            self.send_queue.task_done()

    
    def process_flush(self):
        self.send_queue.join()

        self.clients[self.current_language].send_flush() # Flush active client
            



ROUTER: ClientRouter = None


def stop_gracefully(signal, frame):
    print("Stopping clients")
    if not ROUTER is None:
        ROUTER.close()
    print("Stopped clients")
    exit(0)

signal.signal(signal.SIGINT, stop_gracefully)



def processing_finalize_callback():
    print("INFO in processing finalize callback")
    #TODO teardown maybe here

def processing_error_callback():
    print("INFO In processing error callback")
    #TODO cleanup

def processing_break_callback():
    print("INFO in processing break callback")
    #TODO cleanup

def init_callback():
    print("INFO in processing init callback ")

    global ROUTER

    if not ROUTER is None:
        ROUTER.close()

    ROUTER = ClientRouter(SAMPLE_RATE, MODELFILE, MAPPING)



#TODO move to router
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

    global count
    ilen = stop_int - start_int

    print(f"Text: {text}\nStart: {start}\nStop: {stop}\nCreator: {creator}\nFingerprint: {fingerprint}")
    m_cloud_w.send_packet_result_async(count, count+ilen, [text], 1)
    count += ilen
    #TODO parse and recreate word token array



def data_callback(i, sampleA):
    sample = np.array(sampleA, dtype=np.int16)
    lid_logger.debug(f"sample: {type(sample)} {sample.shape} {sample.min()} {sample.max()}")

    ROUTER.process_audio(sample)
#[251, 260, 290, 298, 306, 315, 284, 294, 291, 371] byteorder='little'



    


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

            ROUTER.process_flush()

            m_cloud_w.send_flush()

            print("WORKER INFO received flush message ==> waiting for packages.")
            MCloud.mcloudpacketdenit(packet)
        elif packet.packet_type() == 4:  # MCloudDone

            raise NotImplemented

            m_cloud_w.wait_for_finish(1, "processing")

            
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            print("WOKRER INFO received DONE message ==> waiting for clients.")
            proceed = False
        elif packet.packet_type() == 5:  # MCloudError

            raise NotImplemented

            # In case of a error or reset message, the processing is stopped immediately by
            # calling m_cloud_wBreak followed by exiting the thread.
            m_cloud_w.wait_for_finish(1, "processing")

            #TODO handle error for clients
            
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            print("WORKER INFO received ERROR message >>> waiting for clients.")
            proceed = False
        elif packet.packet_type() == 6:  # MCloudReset

            raise NotImplemented

            m_cloud_w.stop_processing("processing")

            #TODO handle reset for clients

            print("CLIENT INFO received RESET message >>> waiting for clients.")
            MCloud.mcloudpacketdenit(packet)
            proceed = False
        else:
            print("CLIENT ERROR unknown packet type {!s}".format(packet.packet_type()))


            MCloud.mcloudpacketdenit(packet)
            proceed = False