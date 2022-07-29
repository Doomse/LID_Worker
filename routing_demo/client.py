import logging, signal, time, pathlib, re, numpy as np
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from pythonrecordingclient import session


BASE_DIR = pathlib.Path(__file__).resolve().parent


prc_logger = logging.getLogger('prc')
prc_logger.setLevel(logging.DEBUG)
prc_logger.addHandler(
    logging.FileHandler(BASE_DIR/'client.log', mode='a+')
)


HOST = "i13srv53.ira.uka.de"
C_PORT = 4443


C_NAME = "LID_Client"
C_IN_FINGERPRINT = 'xx'
IN_TYPE = 'audio'
C_OUT_FINGERPRINT = 'xx'
OUT_TYPE = 'text'


CLIENT :session.Session = None

def start_client():
    global CLIENT

    CLIENT = session.Session(
        name=C_NAME,
        input_fingerprint=C_IN_FINGERPRINT,
        input_types=[IN_TYPE],
        output_fingerprint=[C_OUT_FINGERPRINT],
    )
    CLIENT.start(HOST, C_PORT)
    print(f"Successfully created client {C_IN_FINGERPRINT}")

    while not CLIENT.ready:
        time.sleep(0.1)
    print(f"Client {CLIENT.input_fingerprint} is ready")




def stop_gracefully(signal, frame):
    print("Stopping client")
    CLIENT.stop()
    print("Stopped client")
    exit(0)

signal.signal(signal.SIGINT, stop_gracefully)


@session.on_receive
def data_return_callback(cl, data: ET.Element):
    prc_logger.debug(f" <<< {ET.tostring(data, encoding='unicode', method='xml')}")
    text = unquote(data.find('text').text)
    start = data.get('start')
    stop = data.get('stop')
    creator = data.get('creator')
    fingerprint = data.get('fingerprint')
    print(f"\nText: {text}\nStart: {start}\nStop: {stop}\nCreator: {creator}\nFingerprint: {fingerprint}\nPacket:")
    

if __name__ == "__main__":

    while True:
        in_str = input('Packet: ')
        if in_str == 'start':
            start_client()
        elif in_str == 'stop':
            CLIENT.stop()
        elif re.fullmatch(r'[0-9]+(,[0-9]+)*', in_str):
            in_list = in_str.split(',')
            in_arr = np.array( [ int(s) for s in in_list ], dtype=np.int16)
            CLIENT.send_audio(in_arr.data)
        else:
            CLIENT.interface._send_xml(f'<status type="{in_str}"/>')