import scala.io.StdIn.readLine
object lab2 extends App{
  def lev(str1: String, str2: String): Int = {
    val n = str1.length
    val m = str2.length
    val matrix = Array.ofDim[Int](n + 1, m + 1)
    for (i <- 0 to n) {
      matrix(i)(0) = i
    }
    for (j <- 0 to m) {
      matrix(0)(j) = j
    }
    def func(i: Int, j: Int): Int = {
      if (i > n || j > m) {
        matrix(n)(m)
      } else if (str1(i - 1) == str2(j - 1)) {
        matrix(i)(j) = matrix(i - 1)(j - 1)
        func(i + 1, j + 1)
      } else {
        matrix(i)(j) = (matrix(i - 1)(j) + 1).min((matrix(i)(j - 1) + 1)).min(matrix(i - 1)(j - 1) + 1)
        func(i, j + 1)
      }
    }

    func(1, 1)
  }

    val str1 = readLine()
    val str2 = readLine()
    print(lev(str1,str2))

}
