# -*- coding: utf-8 -*-
"""FloodLab — desktop app for the flood consequence model.
Formulae f1-f66 and the system structure are transcribed from the supplied PDF.
The source does not give initial values or a complete numeric series for external drivers;
those are user-editable scenario assumptions, not observations.
"""
import math
import numpy as np

VARS = [
 ('Z1','Число погибших людей'), ('Z2','Длительность поражающего воздействия'),
 ('Z3','Площадь зоны ЧС'), ('Z4','Люди, утратившие имущество первой необходимости'),
 ('Z5','Ущерб коммерческим организациям'), ('Z6','Объём загрязнённого грунта'),
 ('Z7','Площадь земель, выведенных из оборота'), ('Z8','Снижение плодородия земель'),
 ('Z9','Продолжительность аварийного периода'), ('Z10','Продолжительность восстановления'),
 ('Z11','Пострадавшие сельскохозяйственные животные'), ('Z12','Потери урожая'),
 ('Z13','Площадь повреждённых лесов'), ('Z14','Ущерб административной территории')]

def safe_log(x): return math.log10(max(float(x),1e-9))
def pos(x): return max(float(x),1e-9)
def safe_exp(x):
    # Keep the mathematical exp() from the PDF; do not silently cap its argument.
    return math.exp(float(x))
def f(i,z):
    # The auxiliary functions are written in the same numbering as the PDF.
    x=[0.0]+[pos(a) for a in z]
    Z=lambda n:x[n]
    a={
    1:lambda:0.6111*Z(2)**4+0.7217*Z(2)**3+0.7446*Z(2)**2+0.9823*Z(2)+0.0148,
    2:lambda:0.0000057*Z(9)**4-0.00068*Z(9)**3-0.011*Z(9)**2+1.37*Z(9)+224.77,
    3:lambda:0.8985*Z(14)**4+0.176*Z(14)**3+0.7223*Z(14)**2+0.9847*Z(14)+0.927,
    4:lambda:0.4677*Z(7)**4+0.7069*Z(7)**3+0.0868*Z(7)**2+0.1717*Z(7)+0.9457,
    5:lambda:safe_exp(0.8471+0.5422/Z(4)+0.3951*safe_log(Z(4))),
    6:lambda:0.7952*Z(11)+0.3401,
    7:lambda:0.0032*Z(1)**4-1.8*Z(1)**3+336.03*Z(1)**2-24105.24*Z(1)+695052.48,
    8:lambda:0.0077*Z(7)**2+0.9946*Z(7)+0.1374,
    9:lambda:-0.064*Z(9)**4+15.23*Z(9)**3-1140.77*Z(9)**2+24335.32*Z(9)+403947.48,
    10:lambda:0.4986*Z(2)**2+0.0854*Z(2)+0.492,
    11:lambda:0.9716*Z(10)**4+0.7811*Z(10)**3+0.5336*Z(10)**2+0.6221*Z(10)+0.3907,
    12:lambda:-0.14*Z(1)**3+48.83*Z(1)**2-4402.026*Z(1)+283563.063,
    13:lambda:11707.78*Z(3)**(1/8)-1950.012*Z(3)**(1/4)-63.8*Z(3)**0.5-0.11*Z(3)+554.5,
    14:lambda:safe_exp(0.762+0.114/Z(5)+0.3895*safe_log(Z(5))),
    15:lambda:0.0217*Z(7)**2+0.7927*Z(7)+0.4721,
    16:lambda:0.8302*Z(14)**2+0.4483*Z(14)+0.3633,
    17:lambda:0.5845*Z(6)**4+0.8868*Z(6)**3+0.8836*Z(6)**2+0.6114*Z(6)+0.865,
    18:lambda:safe_exp(0.6776+0.9149/Z(8)+0.5615*safe_log(Z(8))),
    19:lambda:safe_exp(0.4224+0.9753/Z(12)+0.1117*safe_log(Z(12))),
    20:lambda:safe_exp(0.9321+0.17/Z(2)+0.347*safe_log(Z(2))),
    21:lambda:safe_exp(0.3229+0.7388/Z(12)+0.2312*safe_log(Z(12))),
    22:lambda:0.9833*Z(4)**4+0.0954*Z(4)**3+0.841*Z(4)**2+0.0554*Z(4)+0.1397,
    23:lambda:safe_exp(0.2076+0.1655/Z(7)+0.6113*safe_log(Z(7))),
    24:lambda:0.4161*Z(9)**4+0.8724*Z(9)**3+0.9778*Z(9)**2+0.2006*Z(9)+0.3202,
    25:lambda:0.5896*Z(13)**2+0.8183*Z(13)+0.1977,
    26:lambda:0.283*Z(10)**2+0.9196*Z(10)+0.1487,
    27:lambda:0.8713*Z(11)**4+0.5895*Z(11)**3+0.0583*Z(11)**2+0.6532*Z(11)+0.4761,
    28:lambda:0.3343*Z(12)**2+0.0034*Z(12)+0.5951,
    29:lambda:safe_exp(0.3608+0.0292/Z(9)+0.4383*safe_log(Z(9))),
    30:lambda:0.5193*Z(6)**4+0.1307*Z(6)**3+0.9083*Z(6)**2+0.4975*Z(6)+0.8935,
    31:lambda:safe_exp(0.5722+0.2981/Z(11)+0.714*safe_log(Z(11))),
    32:lambda:0.0741*Z(4)**4+0.2611*Z(4)**3+0.4477*Z(4)**2+0.4958*Z(4)+0.8284,
    33:lambda:0.5528*Z(10)**2+0.5976*Z(10)+0.7432,
    34:lambda:0.3872*Z(13)**2+0.6341*Z(13)+0.489,
    35:lambda:0.9601*Z(4)**2+0.27*Z(4)+0.1453,
    36:lambda:safe_exp(0.5809+0.2149/Z(2)+0.8216*safe_log(Z(2))),
    37:lambda:0.5853*Z(5)**2+0.7931*Z(5)+0.0132,
    38:lambda:0.6748*Z(10)**4+0.1345*Z(10)**3+0.8504*Z(10)**2+0.9407*Z(10)+0.8215,
    39:lambda:0.0503*Z(2)+0.8566,
    40:lambda:0.4528*Z(4)+0.2391,
    41:lambda:safe_exp(0.5853+0.8867/Z(11)+0.0981*safe_log(Z(11))),
    42:lambda:0.8644*Z(12)**2+0.4174*Z(12)+0.1021,
    43:lambda:0.6815*Z(14)+0.3011,
    44:lambda:0.521*Z(3)**2+0.0578*Z(3)+0.9765,
    45:lambda:safe_exp(0.7846+0.0503/Z(7)+0.4584*safe_log(Z(7))),
    46:lambda:safe_exp(0.2664+0.6038/Z(9)+0.5652*safe_log(Z(9))),
    47:lambda:0.4522*Z(4)**2+0.8492*Z(4)+0.2404,
    48:lambda:0.2115*Z(13)**2+0.3728*Z(13)+0.7422,
    49:lambda:safe_exp(0.0096+0.6482/Z(6)+0.0958*safe_log(Z(6))),
    50:lambda:0.8223*Z(8)**2+0.584*Z(8)+0.2215,
    51:lambda:0.6571*Z(13)**2+0.1468*Z(13)+0.7845,
    52:lambda:0.466*Z(1)**2+0.5152*Z(1)+0.1969,
    53:lambda:0.9482*Z(7)**4+0.5353*Z(7)**3+0.52*Z(7)**2+0.2103*Z(7)+0.5825,
    54:lambda:0.5398*Z(11)**2+0.3594*Z(11)+0.7191,
    55:lambda:0.2407*Z(7)**4+0.0158*Z(7)**3+0.6595*Z(7)**2+0.6349*Z(7)+0.6759,
    56:lambda:safe_exp(0.0963+0.4879/Z(3)+0.8474*safe_log(Z(3))),
    57:lambda:safe_exp(0.0107+0.9191/Z(9)+0.9822*safe_log(Z(9))),
    58:lambda:0.692*Z(2)**2+0.2629*Z(2)+0.1879,
    59:lambda:0.4349*Z(3)+0.6437,
    60:lambda:0.0623*Z(5)**4+0.526*Z(5)**3+0.7183*Z(5)**2+0.9025*Z(5)+0.447,
    61:lambda:0.8509*Z(8)**2+0.7456*Z(8)+0.0468,
    62:lambda:0.8446*Z(11)**4+0.9258*Z(11)**3+0.6296*Z(11)**2+0.5082*Z(11)+0.3431,
    63:lambda:safe_exp(0.4367+0.82/Z(12)+0.803*safe_log(Z(12))),
    64:lambda:0.9985*Z(13)**2+0.8721*Z(13)+0.8347,
    65:lambda:0.3843*Z(7)+0.3849,
    66:lambda:0.7881*Z(9)+0.8325,
    }
    # Do not cap the value of a source formula. If it overflows, the solver should
    # report the numerical problem instead of silently changing the model.
    return float(a[i]())

# Exact structure transcribed from the system on the final page of Наводнения.pdf.
# The ten external drivers are scenario inputs because the source gives no time series.
def rhs(t, y, p):
    raw = np.asarray(y, dtype=float)
    z = np.maximum(raw, 1e-9)  # log/division domain for the source functions
    F = lambda n: f(n, z)
    Rw, Hw, Sem, Vhr, Chr, Sev, Ch, Hr, Vh, Lr = [p[k] for k in ('Rw','Hw','Sem','Vhr','Chr','Sev','Ch','Hr','Vh','Lr')]
    d = np.zeros(14, dtype=float)
    d[0]  = (F(1)*F(2)*Rw - F(3)*(Vhr*Chr*Sev)) / z[0]
    d[1]  = (F(4)*(Rw*Hw) - F(5)*F(6)) / z[1]
    d[2]  = (F(7)*F(8)*F(9)*Sem - F(10)*F(11)*(Rw*Lr)) / z[2]
    d[3]  = (F(12)*F(13)*F(14)*F(15)*F(16) - Rw) / z[3]
    d[4]  = (F(17)*F(18)*F(19) - F(20)*F(21)) / z[4]
    d[5]  = (F(22)*F(23)*F(24)*F(25)*Sem - F(26)*F(27)*F(28)*(Chr*Hr*Vhr)) / z[5]
    d[6]  = (F(29)*(Chr*Hr*Vhr) - (Rw*Sev*Lr)) / z[6]
    d[7]  = (F(30)*F(31)*Sev - F(32)*F(33)*F(34)) / z[7]
    d[8]  = (F(35) - F(36)*F(37)*F(38)) / z[8]
    d[9]  = (F(39)*F(40)*F(41)*F(42)*F(43)*Rw - (Chr*Hr*Vhr)) / z[9]
    d[10] = (F(44)*F(45)*F(46) - F(47)*F(48)) / z[10]
    d[11] = (F(49)*F(50)*F(51)*Sem - F(52)*F(53)*F(54)) / z[11]
    d[12] = (F(55)*(Rw*Sem) - F(56)*F(57)) / z[12]
    d[13] = (F(58)*F(59)*F(60)*F(61)*F(62)*F(63)*F(64) - F(65)*F(66)) / z[13]
    # States are non-negative quantities; at the zero boundary, disallow a negative drift.
    d[(raw <= 1e-9) & (d < 0)] = 0.0
    return d

def scaled_rhs(t, y, p, alpha=1.0):
    """Optional scenario time-scale multiplier; alpha is not in the source PDF."""
    return float(alpha) * rhs(t, y, p)
