from dataclasses import dataclass
import glob
import numpy as np
import os


@dataclass
class ResidualInfo:
    fname: str                  # filename of the residual file
    stream_number: int          # decoding order in the video
    picture_order: int          # raw picture order in GOP
    frame_type: str             # in {"I", "P", "B"}
    display_number:int = -1     # to be filled after sorting


def parse_frame_type(filename:str):
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    frame_part = parts[3]
    if frame_part.startswith(('I', 'P', 'B')):
        return frame_part[0]
    else:
        print(f"Warning: Cannot parse frame type from filename: {filename}")
        return "?"

def parse_stream_number(filename:str):
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    stream_part = parts[1]
    if stream_part.startswith('s'):
        return int(stream_part[1:])
    else:
        print(f"Warning: Cannot parse stream number from filename: {filename}")
        return -1

def parse_picture_order(filename:str):
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    picture_part = parts[2]
    if picture_part.startswith('p'):
        return int(picture_part[1:])
    else:
        print(f"Warning: Cannot parse picture order from filename: {filename}")
        return -1


def get_sorted_residual_info_list(folder:str, space:str) -> list[ResidualInfo]:
    """ Parse and sort filenames in the residual folder (the output folder of JM software) for a specific color space.
        
        The filename components are:
            space: color space, in {"Y","U","V"}
            stream_number: this indicates the decoding order in the video
            picture_order: this indicates the picture order in a GOP, which is used for sorting the display order
            frame_type: the frame type, I-frame, P-frame or B-frame, in {"I","P","B"}
        
        Current nomenclature: img{space}_s{stream_number}_p{picture_order}_{frame_type}.npy

        This nomenclature is defined in the jm h264 decoder tool. 
        
        See function `export_from_inspector` in jm/ldecod/inspect/src/inspect.c

    Parameters
    ----------
    folder : str
        Input folder containing the residual files.
    space : str
        Color space of the residual files, in {"Y","U","V"}.
    """

    pattern = os.path.join(folder, f"img{space}_s*_p*_[I|P|B].npy")
    fnames = glob.glob(pattern)
    frame_info_list = []
    
    for fname in fnames:
        stream_number = parse_stream_number(fname)
        picture_order = parse_picture_order(fname)
        frame_type = parse_frame_type(fname)
        
        frame_info = ResidualInfo(
            fname=fname, 
            stream_number=stream_number,
            picture_order=picture_order,
            frame_type=frame_type
        )

        frame_info_list.append(frame_info)
    
    # make sure every stream number is unique
    stream_numbers = [fi.stream_number for fi in frame_info_list]
    assert len(stream_numbers) == len(set(stream_numbers)), "Error: Duplicate stream numbers found!"

    # step 1: sort by stream number
    frame_info_list.sort(key=lambda x : x.stream_number)

    # step 2: divide into GOPs, reorder picture orders within each GOP
    picture_orders = [fi.picture_order for fi in frame_info_list]

    print("len(picture_orders):", len(picture_orders))

    gop_list = []
    current_gop = [picture_orders[0]]
    for i, po in enumerate(picture_orders[1:], start=1):
        if po > 0:
            current_gop.append(po)
        else:
            # convert the values in current_gop to their ranks
            ranks = np.argsort(np.argsort(current_gop))
            gop_list.append(ranks)
            current_gop = [po]
    ranks = np.argsort(np.argsort(current_gop))
    gop_list.append(ranks)

    # step 3: global reorder according to picture orders in GOPs
    global_picture_orders = []
    picture_order_offset = 0
    for ranks in gop_list:
        global_picture_orders.append(ranks + picture_order_offset)
        picture_order_offset += len(ranks)
    global_picture_orders = np.concatenate(global_picture_orders)
    
    # step 4: sort frame_info_list according to global picture orders
    frame_info_list = np.array(frame_info_list)[np.argsort(global_picture_orders)].tolist()
    for i, fi in enumerate(frame_info_list):
        fi.display_number = i

    return frame_info_list
