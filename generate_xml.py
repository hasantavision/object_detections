import os
import lxml.builder
import lxml.etree

class GenerateXml(object):
    def __init__(self, box_array, im_width, im_height, im_channel, inferred_class, file_name, path, type='jpg'):
        self.inferred_class = inferred_class
        self.box_array = box_array
        self.im_width = im_width
        self.im_height = im_height
        self.file_name = file_name
        self.im_channel = im_channel
        self.path = path
        self.type = type

    def gerenate_basic_structure(self):
        fname = os.path.splitext(self.file_name)
        maker = lxml.builder.ElementMaker()
        xml = maker.annotation(
            maker.folder(),
            maker.filename(fname[0] + ".png" if self.type == 'png' else ".jpg"),
            maker.database(),  # e.g., The VOC2007 Database
            maker.annotation(),  # e.g., Pascal VOC2007
            maker.image(),  # e.g., flickr
            maker.size(
                maker.height(str(self.im_height)),
                maker.width(str(self.im_width)),
                maker.depth(str(self.im_channel)),
            ),
            maker.segmented(),
        )
        
        count = 0
        for box in self.box_array:
            xml.append(
                maker.object(
                    maker.name(self.inferred_class[count]),
                    maker.pose('0'),
                    maker.truncated('0'),
                    maker.difficult('0'),
                    maker.bndbox(
                        maker.xmin(str(box['xmin'])),
                        maker.ymin(str(box['ymin'])),
                        maker.xmax(str(box['xmax'])),
                        maker.ymax(str(box['ymax'])),
                    ),
                )
            )
            count += 1

        with open(os.path.join(self.path, fname[0] + '.xml'), "wb") as f:
            f.write(lxml.etree.tostring(xml, pretty_print=True))


def main():
    # just for debuggind
    xml = GenerateXml([{'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}], '4000', '2000', ['miner', 'miner', 'rust'], 'image_test.xml')
    xml.gerenate_basic_structure()