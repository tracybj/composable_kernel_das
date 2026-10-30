# M128 cooperative and early B experiment

M128 reuses the successful byte-exact cooperative layout with32KiB LDS,64 accumulators/lane and4/8/16 copy-compute waves,N128/N256/N512. Compare late/early B onM128/N256 andearlyB onM256/N512. EarlyB issues next B after nextA copy andbefore low MMAC;the opposite register/LDS slots were retired atprevious stage barrier. Same format1,rawA,INT32,no scale,no16bit API change. Resource/occupancy benefit is a hypothesis;actual residency isnot established bystatic metadata. No production change until numerical andpaired evidence.
