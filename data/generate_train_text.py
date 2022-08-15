import os
import glob

def main():
    print("Start Generate train.txt")
    dir_path = "/home/ekorudiawan/research/DRBox-V2/data/test/*.jpg"
    with open('test.txt', 'w') as f:
        for filename in glob.glob(dir_path):
            #new_filename = filename.lower().replace('(', '_').replace(')','').replace(' ','')
            #os.rename(filename, new_filename)
            label_filename = filename.replace("jpg","rbox")
            if(os.path.exists(label_filename)):
                print(filename)
                f.write(filename + " " + label_filename)
                f.write('\r')
                f.write('\n')
            else:
                print("Label file missing")
                os.remove(filename)

if __name__ == "__main__":
    main()
