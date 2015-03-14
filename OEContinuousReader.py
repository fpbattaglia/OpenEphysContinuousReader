from __future__ import division
__author__ = 'fpbatta'


import numpy as np
import os.path
import re

class ChannelFileReader(object):
    """
    reads a Open Ephys continuous channel file.
    Following the Open Ephys and the Klusters format, times are given in units of
    sampling intervals
    """
    def __init__(self, filename, checkFile=True):
        """
        :param filename:
        """
        self.HEADER_SIZE = 1024
        self.N_SAMPLES_PER_RECORD = 1024
        self.RECORD_SIZE = 2 * self.N_SAMPLES_PER_RECORD + 12 + 10 # data + miniheader + record marker


        self.nRecordingEpochs = None # TODO
        sz = os.path.getsize(filename)

        # make sure that there are not half-full records
        assert ((sz-self.HEADER_SIZE) % self.RECORD_SIZE == 0)

        self.nRecords = int((sz-self.HEADER_SIZE)/self.RECORD_SIZE)

        self.nRecordings = 0
        self.filename = filename

        self.fh = file(self.filename)
        # read the header
        self.header = self._readHeader()

        if checkFile:
            self._checkFile()
        self.fh.seek(self.HEADER_SIZE)
        # do some basic setup
        # something else

    def _readHeader(self):
        """
        :return: a dict with all information in the header
        """
        # 1 kiB header
        header_dt = np.dtype([('Header', 'S%d' % self.HEADER_SIZE)])
        header = np.fromfile(self.fh, dtype=header_dt, count=1)

        # alternative which moves file pointer
        #fid = open(fname, 'rb')
        #header = fid.read(SIZE_HEADER)
        #fid.close()

        # Stand back! I know regex!
        # Annoyingly, there is a newline character missing in the header
        regex = "header\.([\d\w\.\s]{1,}).=.\'*([^\;\']{1,})\'*"
        header_str = str(header[0][0]).rstrip(' ')
        header_dict = {entry[0]: entry[1] for entry in re.compile(regex).findall(header_str)}
        for key in ['bitVolts', 'sampleRate']:
            header_dict[key] = float(header_dict[key])
        for key in ['blockLength', 'bufferSize', 'header_bytes', 'channel']:
            header_dict[key] = int(header_dict[key]) if not key == 'channel' else int(header_dict[key][2:])
        return header_dict

    def _checkFile(self):
        """
        many sanity checks before you start reading the file

        check that timestamps are monotonic, and that difference is 1024
        plus all checks in _readRecord()

        """

        prev_ts = - self.N_SAMPLES_PER_RECORD
        self.fh.seek(self.HEADER_SIZE)
        for i in range(self.nRecords):
            (rec_num, ts, samples) = self._readRecord()
            assert (prev_ts < 0 or (ts-prev_ts) == self.N_SAMPLES_PER_RECORD)
            prev_ts = ts

    def _readRecord(self):
        """
        reads one block of data
        :return:
        """
        data_dt = np.dtype([('timestamp', np.int64),
                            ('n_samples', np.uint16),
                            ('rec_num', np.uint16),
                            # !Note: endianness!
                            ('samples', ('>i2', self.N_SAMPLES_PER_RECORD)),
                            ('rec_mark', (np.uint8, 10))])

        data = np.fromfile(self.fh, dtype=data_dt, count=1)

        # check size of the block

        assert data['samples'].size == self.N_SAMPLES_PER_RECORD
        # check that the record marker is correct
        assert (np.all(data['rec_mark'] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])))

        if data['rec_num'] > self.nRecordings:
            self.nRecordings = data['rec_num']

        return (np.ravel(data['rec_num']),
                np.ravel(data['timestamp']),
                np.ravel(data['samples']))


    def _readBlock(self, nBlocks=1):
        samples = []
        tstamps = []
        rec_num = []
        for i in range(nBlocks):
            (rn, ts, s) = self._readRecord()
            samples.append(s)
            tstamps.append(ts)
            rec_num.append(rn)


        sz = [s.shape for s in samples]
        print sz

        return (np.concatenate(rec_num),
                np.concatenate(tstamps),
                np.concatenate(samples))


    def readBlockRange(self, blockMin, blockMax):
        self.fh.seek(self.HEADER_SIZE + blockMin * self.RECORD_SIZE)
        rt = self._readBlock(blockMax-blockMin)
        return rt

    def readBlocks(self, nBlocks=1):
        rt = self._readBlock(nBlocks)
        yield rt



class ChannelGroup(object):
    pass




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    reader = ChannelFileReader('/Users/fpbatta/Dropbox/DataAnalysisWorkshop/spike_extract_test/100_CH10.continuous')

    print reader.header

    (rn, ts, data) = reader.readBlockRange(0, 100)
    print data.size
    plt.figure(figsize=(13, 5))
    plt.plot(data)
    plt.show()