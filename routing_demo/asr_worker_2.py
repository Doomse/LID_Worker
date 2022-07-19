import time, numpy as np
import MCloud, MCloudPacketRCV #type: ignore


HOST = "i13srv53.ira.uka.de".encode('utf-8')
PORT = 60019

W_NAME = "LID_Worker".encode('utf-8')
IN_FINGERPRINT = 'zz'.encode('utf-8')
IN_TYPE = 'audio'.encode('utf-8')
OUT_FINGERPRINT = 'zz'.encode('utf-8')
OUT_TYPE = 'text'.encode('utf-8')

count = 0


def processing_finalize_callback():
    print("INFO in processing finalize callback")

def processing_error_callback():
    print("INFO In processing error callback")

def processing_break_callback():
    print("INFO in processing break callback")

def init_callback():
    print("INFO in processing init callback ")

def data_callback(i, sampleA):
    print('DEBUG', 'i:', type(i), i)
    print('DEBUG', 'sampleA:', type(sampleA), len(sampleA) )
    sample = np.asarray(sampleA)
    print('DEBUG', 'sample:', type(sample), sample.shape, sample.min(), sample.max() )

    text = ['recieved frames on zz']
    global count
    m_cloud_w.send_packet_result_async(count, count+i, text, len(text))
    count += i
    print('DEBUG', 'count:', type(count), count)
    


m_cloud_w = MCloud.MCloudWrap(W_NAME, 1) # Mode 1: Worker

m_cloud_w.add_service(W_NAME, 'lid'.encode('utf-8'), IN_FINGERPRINT, IN_TYPE, OUT_FINGERPRINT, OUT_TYPE, ''.encode('utf-8'))
m_cloud_w.set_callback("init", init_callback)
m_cloud_w.set_data_callback("worker")
m_cloud_w.set_callback("finalize", processing_finalize_callback)
m_cloud_w.set_callback("error", processing_error_callback)
m_cloud_w.set_callback("break", processing_break_callback)

m_cloud_w.connect(HOST, PORT)


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
            m_cloud_w.send_flush()
            print("WORKER INFO received flush message ==> waiting for packages.")
            MCloud.mcloudpacketdenit(packet)
            break
        elif packet.packet_type() == 4:  # MCloudDone
            print("WOKRER INFO received DONE message ==> waiting for clients.")
            m_cloud_w.wait_for_finish(1, "processing")
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            proceed = False
        elif packet.packet_type() == 5:  # MCloudError
            # In case of a error or reset message, the processing is stopped immediately by
            # calling m_cloud_wBreak followed by exiting the thread.
            m_cloud_w.wait_for_finish(1, "processing")
            m_cloud_w.stop_processing("processing")
            MCloud.mcloudpacketdenit(packet)
            print("WORKER INFO received ERROR message >>> waiting for clients.")
            proceed = False
        elif packet.packet_type() == 6:  # MCloudReset
            m_cloud_w.stop_processing("processing")
            print("CLIENT INFO received RESET message >>> waiting for clients.")
            MCloud.mcloudpacketdenit(packet)
            proceed = False
        else:
            print("CLIENT ERROR unknown packet type {!s}".format(packet.packet_type()))
            proceed = False