<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='linesol' ndim='1'
              e='in fpdtype_t[${str(nupts)}][${str(nexprs)}]'
              wts='in fpdtype_t[${str(nupts_line)}][${str(nupts)}]'
              ls='out fpdtype_t[${str(nupts_line)}][${str(nexprs)}]'
              >
% for ei in range(nexprs):
% for li in range(nupts_line):
ls[${li}][${ei}] = 0.0;
% for ui in range(nupts):
ls[${li}][${ei}] += wts[${li}][${ui}]*e[${ui}][${ei}];
% endfor
% endfor
% endfor
</%pyfr:kernel>
