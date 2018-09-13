
g++ -c  CEC2014.cc 
ar rvs  libkernel_CEC2014.a CEC2014.o
mv -f libkernel_CEC2014.a ../
rm -f *.o