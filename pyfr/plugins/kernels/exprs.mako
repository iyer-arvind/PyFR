<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='exprs' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              e='out fpdtype_t[${str(nexprs)}]'
              >
fpdtype_t rho = u[0];
fpdtype_t invrho = 1.0/u[0];
fpdtype_t rhov[${ndims}];
fpdtype_t v[${ndims}];
fpdtype_t E = u[${nvars - 1}];
% for i in range(ndims):
    rhov[${i}] = u[${i + 1}];
    v[${i}] = invrho*rhov[${i}];
% endfor
fpdtype_t p = ${c['gamma'] - 1}*(E - 0.5*invrho*${pyfr.dot('rhov[{i}]', i=ndims)});

% for i, ex in enumerate(exprs):
e[${i}] = ${ex[1]};
% endfor

</%pyfr:kernel>
