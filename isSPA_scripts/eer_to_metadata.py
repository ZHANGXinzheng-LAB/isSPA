import xmltodict
from tifffile import TiffFile

def eer_to_metadata(eer) -> dict:

    # Read EER and extract data under tag 65001
    with TiffFile(eer) as tif:
        tag = tif.pages[0].tags['65001']
        data = tag.value.decode('UTF-8')

    # Convert the XML to a dict
    parsed = xmltodict.parse(data)

    # Flatten the dict into key:value pairs
    metadata = {}
    for item in parsed["metadata"]["item"]:

        key   = item["@name"]
        value = item["#text"]
        metadata[key] = value

        # If the value has an associated unit
        try:
            unit = item["@unit"]
            metadata[f"{key}.unit"] = unit
        except KeyError:
            pass

    return metadata