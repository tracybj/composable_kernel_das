# Rotated interleaved LDS and staggered consumers

Use the full GPU-byte-verifiedv118 map forinternalM192/N256/K128. Four producers copyinterleaved source rows withm0 wave-tag;A fragment basesarederivedindependently withmodulo1KiB wrap. Raworiginal weightsinternalA/LDS,rawactivationsinternalB/registers. Two32KiB slots andinternalM tailretained.

Compareuniform8versus4+4staggered consumers andMMACpriority0versus1. StaggeredCPwavescapturethehighAhalf beforeitsrelease barrieranddeferthatMMAC tothefollowinglowphase;lastpendinghalf isexplicitlyflushed. CurrentB waitsandLDS waitsusecompiler-visiblebuiltins;verifyactualISAencodings. Format/math/16bitinterfacesunchanged. Keepv117M192andM256stagger controlsinthesamemodule. No performanceclaim before resource/fullrange/tail/odd/even checks andpairedtiming.
