"""Byte-address proof for compact XOR staging and MMAC8-byte fragments."""
def main():
    checks=0
    for rows in [16,32,64,128]:
        for bk in [256,512]:
            shared=[None]*(rows*bk)
            for row in range(rows):
                for ko in range(0,bk,16):
                    address=row*bk+(ko^((row&7)*16))
                    assert address%16==0
                    for b in range(16):
                        assert shared[address+b] is None
                        shared[address+b]=(row,ko+b)
            assert all(x is not None for x in shared)
            for row in range(rows):
                for kk in range(bk//32):
                    for quad in range(4):
                        ko=kk*32+quad*8;address=row*bk+(ko^((row&7)*16))
                        assert address%8==0
                        for b in range(8):
                            assert shared[address+b]==(row,ko+b)
                            checks+=1
    print('COMPACT_XOR_COPY_FRAGMENT_PASS',checks,'byte checks;8 full tiles')

if __name__=='__main__':main()
