import os

def generate_from_directory(direcotry, filename):
    with open(filename + '.txt', 'w') as f:
        for dir_class in os.listdir(direcotry + '/'):
            if os.path.isdir(direcotry + '/' + dir_class) is False:
                continue

            for img in os.listdir(direcotry + '/' + dir_class):
                if "DS" in img:
                    continue

                fullpath = os.path.abspath(direcotry + '/' + dir_class + '/' + img)

                line = dir_class + ':' + fullpath + '\n'
                f.write(line)

generate_from_directory("data", "dataset")
generate_from_directory("test", "test")
