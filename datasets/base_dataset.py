import os, random

class DatumXY:
    """Data instance which defines the basic attributes.

    Args:
        impath_x (str): image path of fake.
        impath_y (str): image path of live.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath_x="", impath_y="", label=-1, domain=-1, classname="", video=""):
        assert isinstance(impath_x, str)
        assert isinstance(impath_y, str)
        self._impath_x = impath_x
        self._impath_y = impath_y
        self._label = label
        self._domain = domain
        self._classname = classname
        self._video = video

    @property
    def impath_x(self):
        return self._impath_x
    @property
    def impath_y(self):
        return self._impath_y
    @property
    def label(self):
        return self._label
    @property
    def domain(self):
        return self._domain
    @property
    def classname(self):
        return self._classname
    @property
    def video(self):
        return self._video

def txt2list(root, protocols, stage):
    # get data from txt
    with open(os.path.join(root, 'Protocol', protocols, stage + '.txt')) as f:
        lines = f.readlines()
        f.close()
    lines_ = []
    for line in lines:
        image, label = line.strip().split(' ')
        if stage == 'train':
            impath = os.path.join(root, image.lstrip('/'))
            lines_.append((impath, int(label)))
        else:
            pairs = []
            pairs.append(os.path.join(root, image.lstrip('/')))
            pairs.append(int(label))
            lines_.append(tuple(pairs))

    # data balance to 1:1
    if stage == 'train':
        lives, fakes = [], []
        for line in lines_:
            impath, label = line
            if label == 1:
                lives.append(line)
            else:
                fakes.append(line)
        insert = len(fakes) - len(lives)
        if insert < 0:
            insert = -insert
            for _ in range(insert):
                fakes.append(random.choice(fakes))
        elif insert > 0:
            for _ in range(insert):
                lives.append(random.choice(lives))
        else:
            pass
        assert len(lives) == len(fakes)
        return lives, fakes 
    else:
        return lines_

def read_data(data_root, input_domain, protocols, split):
    items = []
    if split == 'train':
        lives_list, fakes_list = txt2list(data_root, protocols, split)
        for i in range(len(fakes_list)):
            item = DatumXY(
                impath_x=fakes_list[i][0], 
                impath_y=lives_list[i][0],
                domain=input_domain
            )
            items.append(item)
        print('Load {} {}={} pairs'.format(input_domain, split, len(lives_list)))
        return items
    else:
        val_data_list = txt2list(data_root, protocols, split)
        for impath, label in val_data_list:
            item = DatumXY(
                impath_x=impath,
                label=label
            )
            items.append(item)
        print('Load {} {}={} images'.format(input_domain, split, len(val_data_list)))
        return items

