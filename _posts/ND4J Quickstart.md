

```Scala
%%classpath add mvn 
org.nd4j nd4j-native-platform 0.7.2
org.nd4j nd4s_2.11 0.7.2
```





## Creating an NDArray


```Scala
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
val x1 = Nd4j.create( Array[Double](2,7,6,9,5,1,4,3,8), Array(3,3), 'c')
// or
val x2 = Array(Array(2,7,6),Array(9,5,1),Array(4,3,8)).toNDArray
OutputCell.HIDDEN
```


```Scala
println(x1)
```

    [[2.00, 7.00, 6.00],
     [9.00, 5.00, 1.00],
     [4.00, 3.00, 8.00]]





    null



Notice in the contruction of `x1`, we specify `c`. This will be column major, while `f` would have resulted in row major.


```Scala
val x1 = Nd4j.create( Array[Double](2,7,6,9,5,1,4,3,8), Array(3,3), 'f')
```




    [[2.00, 9.00, 4.00],
     [7.00, 5.00, 3.00],
     [6.00, 1.00, 8.00]]



## Transpose

Very fundamental operation is transposition - this is switching the row index with the column index for each element.


```Scala
println(x1)
println()
println(x1.transpose)
```

    [[2.00, 7.00, 6.00],
     [9.00, 5.00, 1.00],
     [4.00, 3.00, 8.00]]
    
    [[2.00, 9.00, 4.00],
     [7.00, 5.00, 3.00],
     [6.00, 1.00, 8.00]]





    null



One thing of particular importance to point out, in ND4J, transpose is one of several operations that return a reference to the original object.  You are NOT creating a new one. That means if `x` is an $n \times k$
 matrix, and you create `xt = x.transpose` and set `xt(0,1) = 10`, then `x(1,0)` will be set to 10 as well. We’ll see the other operations that act similarly when we discuss slicing.

## Slicing



The fundamental methods of slicing in ND4J involve 'get', 'getRow', 'getRows', 'getColumn', and lastly, 'getColumns'.

'get', 'getRow', 'getColumn' all return objects that point to the same underlying data in memory, same as transpose does. This can be very useful, but you must remain mindful.

There are some other methods of slicing that are specific to Scala (available via ND4S).


```Scala
val rowsToGet = List(1,2)
val xSubSet = x1.getRows( rowsToGet:_*)
println(xSubSet)
```

    [[9.00, 5.00, 1.00],
     [4.00, 3.00, 8.00]]





    null



Some convenient ND4S specific slicing methods


```Scala
println( x1(1 to 2, 0 until 3))
```

    [[9.00, 5.00, 1.00],
     [4.00, 3.00, 8.00]]





    null




```Scala
println( x1(1 to 2, ->))
```

    [[9.00, 5.00, 1.00],
     [4.00, 3.00, 8.00]]





    null



## Setting values

You would set elements of an NDArray in a familiar manner, i.e., `x(0,0) = 10` would set the 0, 0 element equal to 10.


```Scala
// makes a copy of x1 and sets every element equal to 1
// where as x1(->,->) = 1 sets every element of x1 equal to 1
x1.dup(->, ->) = 1

```




    [[1.00, 1.00, 1.00],
     [1.00, 1.00, 1.00],
     [1.00, 1.00, 1.00]]




```Scala
x1
```




    [[2.00, 7.00, 6.00],
     [9.00, 5.00, 1.00],
     [4.00, 3.00, 8.00]]




```Scala
val row1 = x1.getRow(1)
println("row 1")
println(row1)
row1(0,0) = 10
println("\nmodified row 1")
println(row1)
println("\noriginal INDArray that row 1 came from")
println(x1)
```

    row 1
    [10.00, 5.00, 1.00]
    
    modified row 1
    [10.00, 5.00, 1.00]
    
    original INDArray that row 1 came from
    [[2.00, 7.00, 6.00],
     [10.00, 5.00, 1.00],
     [4.00, 3.00, 8.00]]





    null



##  Matrix Mutliplication


```Scala
val y = Nd4j.randn(3,2)

println(y.transpose mmul y)
```

    [[5.77, 5.45],
     [5.45, 11.33]]





    null



## Hadamard Product


```Scala
println( y * y)
```

    [[-4.12, -6.29],
     [2.46, -1.77],
     [-0.14, -1.62]]





    null



INDArray have elementwise methods availble. Some standards, `add`, `mul`, `sub`, `div`. The argument must be the same size as the matrix or a scalar.

All the operations listed above have an inplace version as well: `addi`, `muli`, `subi`, and `divi`. For example addi would be equivalent to `x += 1`.

More elementwise operations are available in org.nd4j.linalg.ops.transforms.Transforms, for example: `exp`, `log`, etc.


```Scala
// mutliply by scaler

println(x.mul(3))

// multiply inplace by a scaler
println()
x.muli(3)

println(x)
```

    [[2.10, -14.71, 0.78],
     [-1.77, 3.22, 10.27],
     [-0.02, -11.85, -6.07],
     [1.03, -4.24, 7.30]]
    
    [[2.10, -14.71, 0.78],
     [-1.77, 3.22, 10.27],
     [-0.02, -11.85, -6.07],
     [1.03, -4.24, 7.30]]





    null



## Broadcasting

There are specific methods available for this including `addRowVector`, `addColumnVector` as well as the multiplication, division and subtraction analogs.  In place versions are also available, for example `addiRowVector`

## Identity Matrics and Matrix Inversion


```Scala
val i = Nd4j.eye(3)
```




    [[1.00, 0.00, 0.00],
     [0.00, 1.00, 0.00],
     [0.00, 0.00, 1.00]]




```Scala
import org.nd4j.linalg.inverse.InvertMatrix.invert
```




    import org.nd4j.linalg.inverse.InvertMatrix.invert





```Scala
val x = Nd4j.randn(4,3)
val xtx = x.transpose mmul x
val xtxinv = invert(xtx, false)  // the the boolean will do the inversion in place by provide true
```




    [[18.96, 1.97, 0.23],
     [1.97, 0.43, -0.05],
     [0.23, -0.05, 0.44]]




```Scala
xtx mmul xtxinv
```




    [[1.00, 0.00, -0.00],
     [-0.00, 1.00, -0.00],
     [-0.00, -0.00, 1.00]]




```Scala
invert(xtx, true)
println(xtx)
```

    [[18.96, 1.97, 0.23],
     [1.97, 0.43, -0.05],
     [0.23, -0.05, 0.44]]





    null



## Other useful operations


```Scala
// sum columns

x.sum(0)
```




    [1.41, -0.40, 0.50]




```Scala
// sum rows

x.sum(1)
```




    [-1.45, -1.95, 2.93, 1.99]




```Scala
// matrix of ones
val ones = Nd4j.ones(2,2)
```




    [[1.00, 1.00],
     [1.00, 1.00]]




```Scala
// ones like
val onesLikeX = Nd4j.onesLike(x)
```




    [[1.00, 1.00, 1.00],
     [1.00, 1.00, 1.00],
     [1.00, 1.00, 1.00],
     [1.00, 1.00, 1.00]]




```Scala
// matrix of zeros
val zeros = Nd4j.zeros(2,2)
```




    [[0.00, 0.00],
     [0.00, 0.00]]




```Scala
// zeros like 
val zerosLikeX = Nd4j.zerosLike(x)
```




    [[0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00]]




```Scala
// horizontal stacking
val onesAndZeros = Nd4j.hstack(ones, zeros) // or Nd4j.concat(1, ones, zeros)
```




    [[1.00, 1.00, 0.00, 0.00],
     [1.00, 1.00, 0.00, 0.00]]




```Scala
// vertical stacking
val onesAndZeros2 = Nd4j.vstack(ones, zeros) // or Nd4j.concat(0, ones, zeros)
```




    [[1.00, 1.00],
     [1.00, 1.00],
     [0.00, 0.00],
     [0.00, 0.00]]




```Scala
// mean of columns 
x.mean(0)
```




    [0.35, -0.10, 0.12]




```Scala
// mean of rows 
x.mean(1)
```




    [-0.48, -0.65, 0.98, 0.66]


