% feynman_qqgqq.mp -- do not edit, generated automatically by notes.tex
input feynmp
require_RCS_revision "1.30";
beginchar(1, 40*2.84526pt#, 20*2.84526pt#, 0);
"feynmf: 1";
LaTeX_unitlength:=2.84526pt;
subgraph (0, 0, w, h);
vinit;
pickup pencircle scaled thin;
vleft (__i1, __i2);
vright (__o1, __o2);
vconnect ("fermion", __i2, __v1, __i1);
vconnect ("gluon, label=$g$, label.dist=0.08w", __v1, __v2);
vconnect ("dashes", __o1, __v2, __o2);
vlabel ("$\bar {q}_j^s, p$", __i1);
vlabel ("$q^r_i, k$", __i2);
vlabel ("$\tilde {q}_m, k'$", __o2);
vlabel ("$\tilde {q}^*_n, p'$", __o1);
vlabel ("$\mu $", __v1);
vlabel ("$\nu $", __v2);
vdot (__v1);
vdot (__v2);
vfreeze;
vdraw;
endsubgraph;
endchar;
% the end.
end.
endinput;
