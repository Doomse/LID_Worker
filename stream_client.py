import argparse, logging, pathlib, signal, time, wave
import functools, math, random, numpy as np
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
    startoffset = data.get('startoffset')
    stopoffset = data.get('stopoffset')
    stop = data.get('stop')
    creator = data.get('creator')
    fingerprint = data.get('fingerprint')
    print(
f"Text: {text}\n\
Start: {start}\n\
Stop: {stop}\n\
Startoffset: {startoffset}\n\
Stopoffset: {stopoffset}\n\
Creator: {creator}\n\
Fingerprint: {fingerprint}\n"
    )



#TODO compare audio segment lengths to detected language switches



def generate_segments(langs, segments, distribution, path_var: pathlib.Path):

    for seg in random.sample(langs, segments, counts= [ segments ] * len(langs) ):
        length = distribution()
        path = path_var/seg
        files = list( path.rglob('*.wav') )

        while length > 0:
            a = random.choice(files)
            with wave.open(str(a), 'rb') as wf:
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                if ch != 1 or sw != 2 or sr != 16000:
                    continue
                nfr = wf.getnframes()
                alen = min(nfr/sr, length)
                length -= alen
                nframes = int(alen*sr)
                yield wf.readframes(nframes)



def send_data(langs, segments, packetsize, distribution, path):
    buffer = b''
    for seg in generate_segments(langs, segments, distribution, path):
        if buffer is None:
            buffer = seg
        else:
            buffer = buffer + seg
        while len(buffer) > packetsize:
            CLIENT.send_audio( buffer[:packetsize] )
            buffer = buffer[packetsize:]
    CLIENT.send_audio( buffer )


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streams concatenated segments of audio in different languages")
    parser.add_argument('path', type=pathlib.Path, help="The filesystem location of the audios, gets language code appended")
    parser.add_argument('-l', '--lang', action="append", help="The languages to include")
    parser.add_argument('--minlen', type=float, default=10., help="Minimum length per language segment")
    parser.add_argument('--maxlen', type=float, default=30., help="Maximum length per language segment")
    parser.add_argument('-s', '--segments', type=int, default=20, help="Number of segments")
    parser.add_argument('-p', '--packetsize', type=int, default=8000, help="Size of stream packets")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-u', '--uniform', action="store_true", help="Uniform distribution of length")
    group.add_argument('-n', '--normal', action="store_true", help="Normal distribution of length")

    args = parser.parse_args()
    start_client()

    if args.uniform:
        distr = functools.partial(random.uniform, args.minlen, args.maxlen)
    if args.normal:
        mu = (args.maxlen + args.minlen) / 2
        sigma = math.sqrt( (args.maxlen - args.minlen) / 2 )
        distr = functools.partial(random.gauss, mu, sigma)

    send_data(args.lang, args.segments, args.packetsize, distr, args.path)

    CLIENT.send_flush()

    print("Flush complete")

    CLIENT.stop()
