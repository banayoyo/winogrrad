## deriverd from https://github.com/andravin/wincnn ##

# wincnn

A simple python module for computing minimal Winograd convolution algorithms for use with
convolutional neural networks as proposed in [1].

Requirements

+ python: version 2.7.6
+ sympy: version 1.0 (0.7.4.1 does not work)

## Example: F(2,3)

For F(m,r) you must select m+r-2 polynomial interpolation points.

In this example we compute transforms for F(2,3) or
F(2x2,3x3) using polynomial interpolation points (0,1,-1).

```
andrew@broadwell:~/develop/wincnn$ python
Python 2.7.11+ (default, Apr 17 2016, 14:00:29) 
[GCC 5.3.1 20160413] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import wincnn
>>> wincnn.showCookToomFilter((0,1,-1), 2, 3)
AT = 
‚é?  1  1   0‚é?
‚é?          ‚é?
‚é?  1  -1  1‚é?

G = 
‚é?1    0     0 ‚é?
‚é?             ‚é?
‚é?/2  1/2   1/2‚é?
‚é?             ‚é?
‚é?/2  -1/2  1/2‚é?
‚é?             ‚é?
‚é?0    0     1 ‚é?

BT = 
‚é?  0   -1  0‚é?
‚é?           ‚é?
‚é?  1   1   0‚é?
‚é?           ‚é?
‚é?  -1  1   0‚é?
‚é?           ‚é?
‚é?  -1  0   1‚é?

AT*((G*g)(BT*d)) =
‚é°d[0]‚ãÖg[0] + d[1]‚ãÖg[1] + d[2]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é£d[1]‚ãÖg[0] + d[2]‚ãÖg[1] + d[3]‚ãÖg[2]‚é?

```

The last matrix is the 1D convolution F(2,3) computed using the
transforms AT, G, and BT, on 4 element signal d[0..3] and 3 element
filter g[0..2], and serves to verify the correctness of the
transforms. This is a symbolic computation, so the result should be
exact.

## Example: F(4,3)

The following example computes transforms for F(4,3).

```
>>> wincnn.showCookToomFilter((0,1,-1,2,-2), 4, 3)
AT = 
‚é?  1  1   1  1   0‚é?
‚é?                 ‚é?
‚é?  1  -1  2  -2  0‚é?
‚é?                 ‚é?
‚é?  1  1   4  4   0‚é?
‚é?                 ‚é?
‚é?  1  -1  8  -8  1‚é?

G = 
‚é?/4     0     0  ‚é?
‚é?                ‚é?
‚é?1/6  -1/6   -1/6‚é?
‚é?                ‚é?
‚é?1/6   1/6   -1/6‚é?
‚é?                ‚é?
‚é?/24  1/12   1/6 ‚é?
‚é?                ‚é?
‚é?/24  -1/12  1/6 ‚é?
‚é?                ‚é?
‚é?0      0     1  ‚é?

BT = 
‚é?  0   -5  0   1  0‚é?
‚é?                  ‚é?
‚é?  -4  -4  1   1  0‚é?
‚é?                  ‚é?
‚é?  4   -4  -1  1  0‚é?
‚é?                  ‚é?
‚é?  -2  -1  2   1  0‚é?
‚é?                  ‚é?
‚é?  2   -1  -2  1  0‚é?
‚é?                  ‚é?
‚é?  4   0   -5  0  1‚é?

AT*((G*g)(BT*d)) =
‚é°d[0]‚ãÖg[0] + d[1]‚ãÖg[1] + d[2]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é¢d[1]‚ãÖg[0] + d[2]‚ãÖg[1] + d[3]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é¢d[2]‚ãÖg[0] + d[3]‚ãÖg[1] + d[4]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é£d[3]‚ãÖg[0] + d[4]‚ãÖg[1] + d[5]‚ãÖg[2]‚é?
```
## Linear Convolution

If instead of an FIR filter you want the algorithm for linear convolution, all you have to do is exchange and transpose the data and inverse transform matrices. This is referred to as the Transfomation Principle.

```
>>> wincnn.showCookToomConvolution((0,1,-1),2,3)
A = 
‚é?  0 ‚é?
‚é?    ‚é?
‚é?  1 ‚é?
‚é?    ‚é?
‚é?  -1‚é?
‚é?    ‚é?
‚é?  1 ‚é?

G = 
‚é?1    0     0 ‚é?
‚é?             ‚é?
‚é?/2  1/2   1/2‚é?
‚é?             ‚é?
‚é?/2  -1/2  1/2‚é?
‚é?             ‚é?
‚é?0    0     1 ‚é?

B = 
‚é?   0  0   0 ‚é?
‚é?            ‚é?
‚é?   1  -1  -1‚é?
‚é?            ‚é?
‚é?1  1  1   0 ‚é?
‚é?            ‚é?
‚é?   0  0   1 ‚é?

Linear Convolution: B*((G*g)(A*d)) =
‚é?     d[0]‚ãÖg[0]      ‚é?
‚é?                    ‚é?
‚é¢d[0]‚ãÖg[1] + d[1]‚ãÖg[0]‚é?
‚é?                    ‚é?
‚é¢d[0]‚ãÖg[2] + d[1]‚ãÖg[1]‚é?
‚é?                    ‚é?
‚é?     d[1]‚ãÖg[2]      ‚é?
```

## Example: F(6,3)

This example computes transform for F(6,3). We will use fraction interpolation points 1/2
and -1/2, so we use sympy.Rational in order to keep the symbolic computation exact (using floating point values would make the derivation of the transforms subject to rounding error).

```
>>> from sympy import Rational
>>> wincnn.showCookToomFilter((0,1,-1,2,-2,Rational(1,2),-Rational(1,2)), 6, 3)
AT = 
‚é?  1  1   1    1    1      1    0‚é?
‚é?                                ‚é?
‚é?  1  -1  2   -2   1/2   -1/2   0‚é?
‚é?                                ‚é?
‚é?  1  1   4    4   1/4    1/4   0‚é?
‚é?                                ‚é?
‚é?  1  -1  8   -8   1/8   -1/8   0‚é?
‚é?                                ‚é?
‚é?  1  1   16  16   1/16  1/16   0‚é?
‚é?                                ‚é?
‚é?  1  -1  32  -32  1/32  -1/32  1‚é?

G = 
‚é?1      0     0  ‚é?
‚é?                ‚é?
‚é?2/9  -2/9   -2/9‚é?
‚é?                ‚é?
‚é?2/9   2/9   -2/9‚é?
‚é?                ‚é?
‚é?/90  1/45   2/45‚é?
‚é?                ‚é?
‚é?/90  -1/45  2/45‚é?
‚é?                ‚é?
‚é?32    16        ‚é?
‚é?‚îÄ‚îÄ    ‚îÄ‚îÄ    8/45‚é?
‚é?45    45        ‚é?
‚é?                ‚é?
‚é?32   -16        ‚é?
‚é?‚îÄ‚îÄ   ‚îÄ‚îÄ‚îÄ‚îÄ   8/45‚é?
‚é?45    45        ‚é?
‚é?                ‚é?
‚é?0      0     1  ‚é?

BT = 
‚é?   0    -21/4    0    21/4     0    -1  0‚é?
‚é?                                         ‚é?
‚é?   1      1    -17/4  -17/4    1    1   0‚é?
‚é?                                         ‚é?
‚é?   -1     1    17/4   -17/4   -1    1   0‚é?
‚é?                                         ‚é?
‚é?  1/2    1/4   -5/2   -5/4     2    1   0‚é?
‚é?                                         ‚é?
‚é?  -1/2   1/4    5/2   -5/4    -2    1   0‚é?
‚é?                                         ‚é?
‚é?   2      4    -5/2    -5     1/2   1   0‚é?
‚é?                                         ‚é?
‚é?   -2     4     5/2    -5    -1/2   1   0‚é?
‚é?                                         ‚é?
‚é?   -1     0    21/4     0    -21/4  0   1‚é?

AT*((G*g)(BT*d)) =
‚é°d[0]‚ãÖg[0] + d[1]‚ãÖg[1] + d[2]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é¢d[1]‚ãÖg[0] + d[2]‚ãÖg[1] + d[3]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é¢d[2]‚ãÖg[0] + d[3]‚ãÖg[1] + d[4]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é¢d[3]‚ãÖg[0] + d[4]‚ãÖg[1] + d[5]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é¢d[4]‚ãÖg[0] + d[5]‚ãÖg[1] + d[6]‚ãÖg[2]‚é?
‚é?                                ‚é?
‚é£d[5]‚ãÖg[0] + d[6]‚ãÖg[1] + d[7]‚ãÖg[2]‚é?
```

[1] "Fast Algorithms for Convolutional Neural Networks" Lavin and Gray, CVPR 2016.
http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf
