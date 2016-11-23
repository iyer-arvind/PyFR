<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='linesol' ndim='1'
              e='in fpdtype_t[${str(nupts)}][${str(nexprs)}]'
              wts='in fpdtype_t[${str(nupts_line)}][${str(nupts)}]'
              ls='out fpdtype_t[${str(nupts_line)}][${str(nexprs)}]'
              >


for(int li=0;li<${nupts_line};li++)
{
% for ei in range(nexprs):
ls[li][${ei}] = 0.0;
% endfor


for(int ui=0; ui<${nupts}; ui++)
{
% for ei in range(nexprs):
ls[li][${ei}]  += wts[li][ui]*e[ui][${ei}];
% endfor
}
}
</%pyfr:kernel>
