
import hashlib 

_hash=hashlib.md5
__handlers={}

def defaulthsh(f):
    def ff(obj,hsh=None):
        if(hsh==None):
            hsh=_hash()
        hsh.update(str(obj.__class__).encode('utf-8'))
        return f(obj,hsh)
    return ff
def handles(typ):
    def fff(f):
        def ff(obj,hsh=None):
            assert (type(obj)==typ),"Invalid type: %s expected %s"%(str(type(obj)),str(typ))
            return f(obj,hsh)
        __handlers[typ]=ff
        return ff
    return fff




@defaulthsh
@handles(list)
def _hashlist(lst,hsh):
    for i in lst:
        hashobject(i,hsh)
    return hsh

@defaulthsh
@handles(tuple)
def _hashtuple(tpl,hsh):
    for i in tpl:
        hashobject(i,hsh)
    return hsh

@defaulthsh
@handles(set)
def _hashset(st,hsh):
    hashlist(sorted(list(st),key=lambda x:str(x)))
    return hsh

@defaulthsh
@handles(dict)
def _hashdict(dct,hsh):
    for k in sorted(dct.keys(),key=lambda x:str(x)):
        hashobject(k,hsh)
        hashobject(dct[k],hsh)
    return hsh

@defaulthsh
@handles(str)
def _hashstr(strg,hsh):
    hsh.update(strg.encode('utf-8'))
    return hsh

@defaulthsh
@handles(int)
def _hashint(i,hsh):
    hsh.update((i).to_bytes(2,'big'))
    return hsh
        
@defaulthsh
@handles(float)
def _hashfloat(f,hsh):
    hsh.update(f.hex().encode('utf-8'))
    return hsh


@defaulthsh
def hashobject(obj,hsh):
    if(type(obj) in __handlers):
        return __handlers[type(obj)](obj,hsh)
    else:
        raise ValueError

def _test():
    for i in [1,
            1.0,
            "1.0",
            [1,2,3,4.0,"5"],
            (1,2,3,4.0,"5"),
            {"a":7,"b":8.7,8.9:{8:"eight",9:"nine"}},
            {"a":7,"b":8.7,8.9:{8:"eight",9:"ninea"}},
            {"a":7,"b":8.8,8.9:{8:"eight",9:"ninea"}},
            {"b":8.7,8.9:{8:"eight",9:"nine","a":7}}
            ]:
        print(hashobject(i).hexdigest())
if(__name__=="__main__"):
    _test()
