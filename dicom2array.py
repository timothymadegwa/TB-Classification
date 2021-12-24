import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    """ Convert dicom file to numpy array 
    
    Args:
        path (str): Path to the dicom file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply monochrome fix
        
    Returns:
        Numpy array of the respective dicom file 
        
    """
    # Use the pydicom library to read the dicom file
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to 
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    # The XRAY may look inverted
    #   - If we want to fix this we can
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    # Normalize the image array and return
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

    def plot_image(img, title="", figsize=(8,8), cmap=None):
    """ Function to plot an image to save a bit of time """
    plt.figure(figsize=figsize)
    
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        img
        plt.imshow(img)
        
    plt.title(title, fontweight="bold")
    plt.axis(False)
    plt.show()
    
def get_image_id(path):
    """ Function to return the image-id from a path """
    return path.rsplit("/", 1)[1].rsplit(".", 1)[0]
