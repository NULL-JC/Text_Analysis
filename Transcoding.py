import codecs

def handleEncoding(original_file, newfile):
    source_encoding='GB2312'
    #####按照确定的encoding读取文件内容，并另存为utf-8编码：
    block_size = 1024
    with codecs.open(original_file, 'r', source_encoding, errors='ignore') as f:
        with codecs.open(newfile, 'w', 'utf-8') as f2:
            while True:
                content = f.read(block_size)
                if not content:
                    break
                f2.write(content)

handleEncoding("data/nCov_10k_test.csv", "data/nCov_10k_test_u.csv")
handleEncoding("data/nCoV_100k_train.labled.csv", "data/nCoV_100k_train.labled_u.csv")


