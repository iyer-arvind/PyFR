import tarfile
import pyfr
import os
import io
import hashlib
import pyfr.hashobject as ho


def _filechksum(f):
    hsh=hashlib.sha256()
    with open(f) as fh:
        hsh.update(fh.read().encode())
    return hsh.hexdigest()


class SourceCode(object):
    def __init__(self):
        self.sums={}

    def createArchive(self):
        base=(os.path.dirname(os.path.abspath(pyfr.__file__)))
        fList=[]
        accept=('.mako','.py','.txt')
        for root, dirs, files in os.walk(base):
            fList.extend(map(lambda x:os.path.join(base,os.path.join(root,x)),filter(lambda x:os.path.splitext(x)[1] in accept,files)))

        header={}
        for f in fList:
            arcname=os.path.relpath(f,base)
            chksum=_filechksum(f)
            header[arcname]=chksum
        
        header['__all__']=ho.hashobject(header,hashlib.sha256()).hexdigest()

        buf=io.BytesIO()
        tar=tarfile.open(name='mem',fileobj=buf,mode="w:gz",format=tarfile.PAX_FORMAT,pax_headers=header)

        def ffilter(tin):
            tout=tin
            tout.pax_headers.update({'chksum':header[tout.name]})
            return tout

        for f in fList:
            arcname=os.path.relpath(f,base)
            tar.add(name=f,arcname=os.path.relpath(f,base),filter=ffilter)

        self.__hash=header.pop('__all__')
        self.__sums=header

        self.__buf=buf

    def load(self,data):
        pass

    def diff(self):
        pass

    def hash(self):
        return self.__hash

    def raw(self):
        return self.__buf.getvalue()

    def size(self):
        self.__buf.seek(0, os.SEEK_END)
        return self.__buf.tell()



if(__name__=="__main__"):
    s=SourceCode()
    with open('tar.tar.gz','wb') as fh:
        fh.write(s.createArchive())
    print (s.hash)
