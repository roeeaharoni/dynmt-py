import random
import codecs

# generate toy parallel sequences: characters to numbers
def main():
    lines = 1000
    input_file_path = '../data/input.txt'
    output_file_path = '../data/output.txt'
    with codecs.open(input_file_path, 'w', encoding='utf8') as input:
        with codecs.open(output_file_path, 'w', encoding='utf8') as output:
            for i in xrange(lines):
                input_line = ''
                output_line = ''
                length = random.randrange(1, 30)
                for j in xrange(length):
                    char_index = random.randrange(97, 122)
                    char = chr(char_index)
                    input_line += char + ' '
                    output_line += str(char_index) + ' '
                output.write(output_line + '\n')
                input.write(input_line + '\n')


if __name__ == '__main__':
    main()