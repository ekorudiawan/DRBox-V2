import numpy as np
import glob

dir_path = "/home/barelangfc/research/DRBox-V2/data/train/*.rbox"

def main():
    list_l = []
    list_w = []
    for filename in glob.glob(dir_path):
        #print("Filenama : ", filename)
        with open(filename, 'r') as f:
            list_rows = f.readlines()
            for row in list_rows:
                #print("Row :", row)
                list_str = row.split(' ')
                l = list_str[2]
                w = list_str[3]
                #print("l :", l)
                #print("w :", w)
                list_l.append(float(l))
                list_w.append(float(w))
    print("annotations statistics")
    print("Length")
    print("min l :", int(np.asarray(list_l).min()))
    print("med l :", int(np.median(np.asarray(list_l))))
    print("max l :", int(np.asarray(list_l).max()))
    print("Width")
    print("min w :", int(np.asarray(list_w).min()))
    print("med w :", int(np.median(np.asarray(list_w))))
    print("max w :", int(np.asarray(list_w).max()))

if __name__ == "__main__":
    main()
