"""K permutation and existing prepared-byte layout proof,not GPU validation."""
def main():
    order=[q*16+p*8+b for p in range(2) for q in range(4) for b in range(8)]
    assert sorted(order)==list(range(64))
    def av(row,k):return ((row*37+k*17+7)%256)-128
    def bv(row,k):return ((row*13+k*29+31)%256)-128
    checked=0
    for n in [256,512,1792,4096]:
        for k in [512,768,8192]:
            for row in [0,1,15]:
                for col in [0,15,16,n-1]:
                    reference=sum(av(row,x)*bv(col,x) for x in range(k));actual=0
                    for kt in range(k//64):
                        for x in order:
                            lane=(x//16)*16+col%16;byte=x%16
                            address=((kt*(n//16)+col//16)*64+lane)*16+byte
                            # Decode independently from[K64,N16,Q4,Row16,Byte16].
                            z,bb=divmod(address,16);z,rr=divmod(z,16);z,qq=divmod(z,4);kk,nn=divmod(z,n//16)
                            assert (nn*16+rr,kk*64+qq*16+bb)==(col,kt*64+x)
                            actual+=av(row,kt*64+x)*bv(nn*16+rr,kk*64+qq*16+bb);checked+=1
                    assert reference==actual
    print('PACKED_K64_DOT_LAYOUT_PASS',checked,'byte/pair contributions;144 exact dots')

if __name__=='__main__':main()
