import tables
import numpy
from keras.utils.io_utils import HDF5Matrix


def test1():
    s1 = numpy.array([['a'*2, 'b'*4], ['a'*2, 'b'*4]], dtype='S8')
    s2 = numpy.array([['a'*6, 'b'*8], ['a'*2, 'b'*4]], dtype='S8')
    s3 = numpy.array([['c'*10, 'd'*1], ['a'*2, 'b'*4]], dtype='S8')

    fileh = tables.open_file('earray1.h5', mode='w')

    array_c = fileh.create_earray(fileh.root, 'array_c', tables.Atom.from_dtype(s1.dtype), (0,2,2))

    array_c.append(s1.reshape(-1,2,2))
    array_c.append(s2.reshape(-1,2,2))
    array_c.append(s3.reshape(-1,2,2))

    # Read the string ``EArray`` we have created on disk.
    for s in array_c:
        print('array_c[%s] => %r' % (array_c.nrow, s))
    # Close the file.
    fileh.close()

def test2():
    fh = tables.open_file('../build/disjoint/stupid_simple_1s_full_5s_window.h5', mode='r')
    print(fh)
    print("HD5F Input: ")
    print(fh.root.input_matrices[:])

    print("HD5F Output: ")
    print(fh.root.output_matrices[:])
    fh.close()

def test3():
    fh = tables.open_file('../build/disjoint/stupid_simple_1s_full_5s_window.h5', mode='r')
    print(fh)
    try:
        print(fh.root.not_real[:])
    except tables.exceptions.NoSuchNodeError:
        print("node doesn't exist")
    print(fh.get_node('/input_matrixes')[:])
    fh.close()

def test4():
    fh = tables.open_file('../build/overlapping/stupid_simple_1s_full_5s_window.h5', mode='r')
    print(fh)
    print("HD5F Input: ")
    print(fh.root.input_matrices[:])

    print("HD5F Output: ")
    print(fh.root.output_matrices[:])
    fh.close()

def test5():
    fh1 = tables.open_file('../build/overlapping/stupid_simple_1s_full_5s_window.h5', mode='r')
    fh1_in = fh1.root.input_matrices[:]
    fh1_out = fh1.root.output_matrices[:]

    fh2 = tables.open_file('../build/overlapping/stupid_simple_1s_full_5s_window_check.h5', mode='r')
    fh2_in = fh2.root.input_matrices[:]
    fh2_out = fh2.root.output_matrices[:]

    in_diff = False
    for i in range(len(fh1_in)):
        for j in range(len(fh1_in[0])):
            for k in range(len(fh1_in[0][0])):
                if fh1_in[i][j][k] != fh2_in[i][j][k]:
                    in_diff = True

    out_diff = False
    for i in range(len(fh1_out)):
        for j in range(len(fh1_out[0])):
            for k in range(len(fh1_out[0][0])):
                if fh1_out[i][j][k] != fh2_out[i][j][k]:
                    out_diff = True

    print("Same input: " + str(not in_diff))
    print("Same output: " + str(not out_diff))

    fh1.close()
    fh2.close()

def test6():
    train_filename = '../build/overlapping/stupid_simple_1s_full_5s_window_check.h5'
    input_group = 'input_matrices'
    output_group = 'output_matrices'
    x_train = HDF5Matrix(train_filename, input_group)
    y_train = HDF5Matrix(train_filename, output_group)
    print(x_train[0])
    print(len(x_train))

def test7():
    fh = tables.open_file('../build/lights-only-overlapping/stupid_simple_1s_full_15s_window.h5', mode='r')
    print(fh)
    print("HD5F Input: ")
    print(fh.root.input_matrices[:5])

    print("HD5F Output: ")
    print(fh.root.output_matrices[:5])
    fh.close()

test7()
