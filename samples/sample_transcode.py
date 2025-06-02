import python_vali as vali
import numpy as np
import av

input_file = "../tests/data/test.mp4"
output_file = 'output.mp4'

target_width = 640
target_height = 360

# GPU-accelerated decoder
py_decoder = vali.PyDecoder(
    input_file,
    {},
    gpu_id=0)

# GPU-accelerated resizer
py_resizer = vali.PySurfaceResizer(py_decoder.Format, gpu_id=0)

# GPU-accelerated encoder
nv_encoder = vali.PyNvEncoder(
    {
        "preset": "P4",
        "codec": "h264",
        "s": str(target_width) + 'x' + str(target_height),
        "bitrate": "1M",
        "fps": f"{py_decoder.Framerate}",
    },
    gpu_id=0
)

# Muxer
dst_file = av.open(output_file, 'w')
out_stream = dst_file.add_stream('h264', rate=int(py_decoder.Framerate))
out_stream.width = target_width
out_stream.height = target_height

def make_packet(
        video_packet: np.ndarray,
        pkt_data: vali.PacketData,
        out_stream: av.VideoStream) -> av.Packet:
    """Create an AV packet from encoded video data and packet metadata.

    This function takes encoded video data and associated metadata to create a properly
    formatted AV packet that can be written to an output video stream.

    Args:
        video_packet (np.ndarray): A numpy array containing the encoded video packet data
        pkt_data (vali.PacketData): Packet metadata containing presentation and decode timestamps
        out_stream (av.VideoStream): The output video stream to associate with the packet

    Returns:
        av.Packet: A properly formatted AV packet ready for muxing into the output stream
    """

    pkt = av.packet.Packet(bytearray(video_packet))
    pkt.stream = out_stream
    pkt.pts = pkt_data.pts
    pkt.dts = pkt_data.dts

    return pkt

# Decoded Surface
surf_src = vali.Surface.Make(
    format=py_decoder.Format,
    width=py_decoder.Width,
    height=py_decoder.Height,
    gpu_id=0)

# Resized Surface
surf_dst = vali.Surface.Make(
    format=py_decoder.Format,
    width=target_width,
    height=target_height,
    gpu_id=0
)

# Numpy array which contains encoded packet
packet = np.ndarray(
    dtype=np.uint8,
    shape=())

pkt_data = vali.PacketData()

while True:
    # Decode single Surface
    success, details = py_decoder.DecodeSingleSurface(surf_src, pkt_data)
    if not success:
        print(details)
        break

    # Resize
    success, details = py_resizer.Run(surf_src, surf_dst)
    if not success:
        print(details)
        break

    # Encode. If there's a compressed packet, write to file
    got_pkt = nv_encoder.EncodeSingleSurface(surf_dst, packet)
    if got_pkt:
        av_pkt = make_packet(packet, pkt_data, out_stream)
        dst_file.mux(av_pkt)

# Encoder is async, so we need to flush it in the end
while True:
    got_pkt = nv_encoder.FlushSinglePacket(packet)
    if got_pkt:
        av_pkt = make_packet(packet, pkt_data, out_stream)
        dst_file.mux(av_pkt)
    else:
        break

# Finish pending operations on file and close it
dst_file.close() 