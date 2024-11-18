from sympy.parsing.mathematica import parse_mathematica

int_t_M2_thetaintegrand_str = "((-4*Sqrt[t - tm]*Sqrt[t - tp]*Log[2*t - tm + 2*Sqrt[t - tm]*Sqrt[t - \
tp] - tp])/Sqrt[(t - tm)*(t - tp)] + (Sqrt[(t - tm)*(t - tp)]*(((s - \
2*Subscript[m, p]^2)^2*(-4*Subscript[m, d]^2 + Subscript[m, \
p]^2)^2)/((s + t - Subscript[m, d]^2 - 2*Subscript[m, p]^2)*(s + tm - \
Subscript[m, d]^2 - 2*Subscript[m, p]^2)*(s + tp - Subscript[m, d]^2 \
- 2*Subscript[m, p]^2)) + (4*Subscript[m, d]^8 + s^3*Subscript[m, \
p]^2 + 8*Subscript[m, d]^6*(s - 4*Subscript[m, p]^2) - s*Subscript[m, \
d]^2*(s^2 - 4*s*Subscript[m, p]^2 + 16*Subscript[m, p]^4) + \
Subscript[m, d]^4*(5*s^2 - 28*s*Subscript[m, p]^2 + 64*Subscript[m, \
p]^4))/((t - Subscript[m, p]^2)*(-tm + Subscript[m, p]^2)*(-tp + \
Subscript[m, p]^2))))/(-s + Subscript[m, d]^2 + Subscript[m, p]^2)^2 \
+ (Sqrt[t - tm]*Sqrt[t - tp]*(((Log[s + t - Subscript[m, d]^2 - \
2*Subscript[m, p]^2] - Log[-2*s*t + s*tm - t*tm + s*tp - t*tp + \
2*tm*tp + (2*t - tm - tp)*Subscript[m, d]^2 + (4*t - 2*(tm + \
tp))*Subscript[m, p]^2 + 2*Sqrt[t - tm]*Sqrt[t - tp]*Sqrt[s + tm - \
Subscript[m, d]^2 - 2*Subscript[m, p]^2]*Sqrt[s + tp - Subscript[m, \
d]^2 - 2*Subscript[m, p]^2]])*(s - 2*Subscript[m, p]^2)*(2*s^3*(s + \
tm)*(s + tp) - 2*s^2*(9*s^2 + 5*tm*tp + 7*s*(tm + tp))*Subscript[m, \
p]^2 + s*(66*s^2 + 20*tm*tp + 39*s*(tm + tp))*Subscript[m, p]^4 - \
(118*s^2 + 8*tm*tp + 45*s*(tm + tp))*Subscript[m, p]^6 + 2*(48*s + \
7*(tm + tp))*Subscript[m, p]^8 - 24*Subscript[m, p]^10 - \
128*Subscript[m, d]^8*(s - 2*Subscript[m, p]^2) + 2*Subscript[m, \
d]^6*(s*(143*s + 56*(tm + tp)) - 4*(129*s + 28*(tm + \
tp))*Subscript[m, p]^2 + 456*Subscript[m, p]^4) - 2*Subscript[m, \
d]^4*(s*(93*s^2 + 48*tm*tp + 71*s*(tm + tp)) - (463*s^2 + 96*tm*tp + \
240*s*(tm + tp))*Subscript[m, p]^2 + 3*(245*s + 64*(tm + \
tp))*Subscript[m, p]^4 - 366*Subscript[m, p]^6) + Subscript[m, \
d]^2*(2*s^2*(13*s^2 + 15*tm*tp + 14*s*(tm + tp)) - 2*s*(66*s^2 + \
28*tm*tp + 49*s*(tm + tp))*Subscript[m, p]^2 + (172*s^2 - 16*tm*tp + \
53*s*(tm + tp))*Subscript[m, p]^4 + (50*s + 54*(tm + \
tp))*Subscript[m, p]^6 - 148*Subscript[m, p]^8)))/(2*(s + tm - \
Subscript[m, d]^2 - 2*Subscript[m, p]^2)^(3/2)*(s + tp - Subscript[m, \
d]^2 - 2*Subscript[m, p]^2)^(3/2)*(s - Subscript[m, d]^2 - \
Subscript[m, p]^2)^3) + ((-Log[t - Subscript[m, p]^2] + Log[-(t*(tm + \
tp)) + (2*t - tm - tp)*Subscript[m, p]^2 + 2*(tm*tp + Sqrt[t - \
tm]*Sqrt[t - tp]*Sqrt[-tm + Subscript[m, p]^2]*Sqrt[-tp + \
Subscript[m, p]^2])])*(4*Subscript[m, d]^10*(tm + tp - 2*Subscript[m, \
p]^2) + 4*Subscript[m, d]^8*(-4*tm*tp + s*(tm + tp) - (2*s + 3*(tm + \
tp))*Subscript[m, p]^2 + 10*Subscript[m, p]^4) + Subscript[m, \
d]^6*(-3*s*(-16*tm*tp + s*(tm + tp)) + (6*s^2 - 32*tm*tp - 36*s*(tm + \
tp))*Subscript[m, p]^2 + 8*(3*s + 8*(tm + tp))*Subscript[m, p]^4 - \
96*Subscript[m, p]^6) - Subscript[m, d]^4*(6*s^2*(-8*tm*tp + s*(tm + \
tp)) + s*(-12*s^2 + 336*tm*tp + 11*s*(tm + tp))*Subscript[m, p]^2 + \
(26*s^2 - 384*tm*tp - 228*s*(tm + tp))*Subscript[m, p]^4 + 40*(3*s + \
8*(tm + tp))*Subscript[m, p]^6 - 256*Subscript[m, p]^8) + \
s*Subscript[m, d]^2*(s^2*(-14*tm*tp + s*(tm + tp)) - 2*s*(s^2 - \
58*tm*tp - 5*s*(tm + tp))*Subscript[m, p]^2 - 6*(s^2 + 24*tm*tp + \
16*s*(tm + tp))*Subscript[m, p]^4 + 4*(19*s + 32*(tm + \
tp))*Subscript[m, p]^6 - 112*Subscript[m, p]^8) + s^2*(-2*s^2*tm*tp + \
s*(-2*tm*tp + s*(tm + tp))*Subscript[m, p]^2 + (8*tm*tp + 3*s*(tm + \
tp))*Subscript[m, p]^4 - 4*(s + 2*(tm + tp))*Subscript[m, p]^6 + \
8*Subscript[m, p]^8)))/(2*(-tm + Subscript[m, p]^2)^(3/2)*(-tp + \
Subscript[m, p]^2)^(3/2)*(-s + Subscript[m, d]^2 + Subscript[m, \
p]^2)^3)))/Sqrt[(t - tm)*(t - tp)])/2"

int_t_M2_thetaintegrand_str_2 = """
-4*Subscript[m, d]^2*(s - 2*Subscript[m, p]^2)*((Sqrt[(t - tm)*(t - tp)]*(((s - 2*Subscript[m, p]^2)*(-2*Subscript[m, d]^2 + Subscript[m, p]^2))/((s + t - Subscript[m, d]^2 - 2*Subscript[m, p]^2)*(s + tm - Subscript[m, d]^2 - 2*Subscript[m, p]^2)*(s + tp - Subscript[m, d]^2 - 2*Subscript[m, p]^2)) + (2*Subscript[m, d]^4 + s*Subscript[m, p]^2 - 4*Subscript[m, d]^2*Subscript[m, p]^2)/((t - Subscript[m, p]^2)*(tm - Subscript[m, p]^2)*(-tp + Subscript[m, p]^2))))/(-s + Subscript[m, d]^2 + Subscript[m, p]^2)^2 + (Sqrt[t - tm]*Sqrt[t - tp]*ArcTan[(Sqrt[t - tm]*Sqrt[-s - tp + Subscript[m, d]^2 + 2*Subscript[m, p]^2])/(Sqrt[t - tp]*Sqrt[s + tm - Subscript[m, d]^2 - 2*Subscript[m, p]^2])]*(s - 2*Subscript[m, p]^2)*(4*s*(s + tm)*(s + tp) - 16*Subscript[m, d]^6 - 7*s*(2*s + tm + tp)*Subscript[m, p]^2 + (10*s - tm - tp)*Subscript[m, p]^4 + 4*Subscript[m, p]^6 + 2*Subscript[m, d]^4*(18*s + 7*(tm + tp) - 29*Subscript[m, p]^2) - Subscript[m, d]^2*(6*(4*s^2 + 2*tm*tp + 3*s*(tm + tp)) - (72*s + 25*(tm + tp))*Subscript[m, p]^2 + 50*Subscript[m, p]^4)))/(Sqrt[(t - tm)*(t - tp)]*(s + tm - Subscript[m, d]^2 - 2*Subscript[m, p]^2)^(3/2)*(s - Subscript[m, d]^2 - Subscript[m, p]^2)^3*(-s - tp + Subscript[m, d]^2 + 2*Subscript[m, p]^2)^(3/2)) - (Sqrt[t - tm]*Sqrt[t - tp]*ArcTan[(Sqrt[t - tm]*Sqrt[-tp + Subscript[m, p]^2])/(Sqrt[t - tp]*Sqrt[tm - Subscript[m, p]^2])]*(2*Subscript[m, d]^6*(tm + tp - 2*Subscript[m, p]^2) - 2*Subscript[m, d]^4*(tm + tp - 2*Subscript[m, p]^2)*(s + Subscript[m, p]^2) + Subscript[m, d]^2*(12*s*tm*tp - (24*tm*tp + 7*s*(tm + tp))*Subscript[m, p]^2 + 2*(s + 10*(tm + tp))*Subscript[m, p]^4 - 16*Subscript[m, p]^6) + s*(-4*s*tm*tp + (8*tm*tp + 3*s*(tm + tp))*Subscript[m, p]^2 - (2*s + 7*(tm + tp))*Subscript[m, p]^4 + 6*Subscript[m, p]^6)))/(Sqrt[(t - tm)*(t - tp)]*(tm - Subscript[m, p]^2)^(3/2)*(-tp + Subscript[m, p]^2)^(3/2)*(-s + Subscript[m, d]^2 + Subscript[m, p]^2)^3))
"""

int_t_M2 = "(-4*t + 4*Subscript[m, p]^2 + (Subscript[m, p]^4*(s - 2*Subscript[m, \
p]^2)^2)/((s + t - Subscript[m, d]^2 - 2*Subscript[m, p]^2)*(-s + \
Subscript[m, d]^2 + Subscript[m, p]^2)^2) - ((s - 2*Subscript[m, \
d]^2)^2*(s*Subscript[m, d]^2 - Subscript[m, d]^4 - s*Subscript[m, \
p]^2))/((t - Subscript[m, p]^2)*(-s + Subscript[m, d]^2 + \
Subscript[m, p]^2)^2) + (Log[s + t - Subscript[m, d]^2 - \
2*Subscript[m, p]^2]*(s - 2*Subscript[m, p]^2)*(s^3 - \
5*s^2*Subscript[m, p]^2 + 10*s*Subscript[m, p]^4 - 4*Subscript[m, \
p]^6 - Subscript[m, d]^2*(s^2 - 4*s*Subscript[m, p]^2 + \
8*Subscript[m, p]^4)))/(s - Subscript[m, d]^2 - Subscript[m, p]^2)^3 \
- (Log[t - Subscript[m, p]^2]*(8*Subscript[m, d]^8 - 8*Subscript[m, \
d]^6*(3*s - 2*Subscript[m, p]^2) + 24*s*Subscript[m, d]^4*(s - \
Subscript[m, p]^2) + s^2*(s^2 + s*Subscript[m, p]^2 - 4*Subscript[m, \
p]^4) + s*Subscript[m, d]^2*(-9*s^2 + 6*s*Subscript[m, p]^2 + \
8*Subscript[m, p]^4)))/(s - Subscript[m, d]^2 - Subscript[m, \
p]^2)^3)/2"

def translate(expr_str):
    expr = str(parse_mathematica(expr_str))
    expr = expr.replace('sqrt','np.sqrt').replace('log', 'np.log').replace('Subscript(m, p)', 'm_phi').replace('Subscript(m, d)', 'm_d').replace('tm','t_m').replace('tp','t_p')
    print(expr)

translate(int_t_M2_thetaintegrand_str_2)

