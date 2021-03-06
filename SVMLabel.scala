package SVM

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark._
import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV,
  axpy => brzAxpy,
  svd => brzSvd,
  max => Bmax,
  min => Bmin,
  sum => Bsum
}
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ArrayBuffer
import java.io._

import scala.math._
import scala.io._
import scala.util.Random


//label目标值，pre预测（分类）结果，err误差
case class SVMLabel(label: BDM[Double], pre: BDM[Double], err: Double) extends Serializable 

case class SVMConfig(
    numVars: Int,
    gamma: Double,
//    sum: Double,
    C: Double,
    eps: Double,
    tolerance: Double,
    train_num: Int) extends Serializable
    
//case class WeightConfig(
//    i: Int,
//    j: Int,
//    initW: Array[BDM[Double]],
//    initA: Array[BDM[Double]],
//    b: Double,
//    error_cache: BDM[Double]) extends Serializable
    
    
class SVMO(
    private var numVars:  Int,
    private var C:  Double,
    private var b:  Double,
//    private var i: Int,
//    private var j: Int,
    private var eps: Double,
    private var gamma: Double,
    private var tolerance: Double,
    private var train_num: Int,
    private var initW: Array[BDM[Double]],
    private var initA: Array[BDM[Double]],
    private var error_cache: BDM[Double]) extends Serializable with Logging{
  
  //初始化参数
  //这部分还可以优化一下，参数配置在对象中重新配置过了
  def this() = this(14,0.1,0.0,0.001,0.01,0.01,1000, Array(BDM.zeros[Double](100,1)),Array(BDM.zeros[Double](100,1)),BDM.zeros[Double](100,1))
  
  def setGamma(gamma: Double): this.type = {
    this.gamma = gamma
    this
  }
  
//  def setb(b: Double): this.type = {
//    this.b = b
//    this
//  }
  
  def setC(C : Double): this.type = {
    this.C = C
    this
  }
  
  def setEps(eps: Double): this.type = {
    this.eps = eps
    this
  }
  
  def setInitW(initW: Array[BDM[Double]]): this.type = {
    this.initW = initW
    this
  }
  
  def setInitA(initA: Array[BDM[Double]]): this.type = {
    this.initA = initA
    this
  }
  
  
  def SVMtrain(train_d: RDD[(BDM[Double],BDM[Double])]): SvmModel = {
    val sc = train_d.sparkContext
    var initStartTime = System.currentTimeMillis()
    var initEndTime = System.currentTimeMillis()
    
    //将数据取出
    var X = train_d.map(f => f._2).collect().apply(0)
    var Y = train_d.map(f => f._1).collect().apply(0)
    
    
    //训练样本数量
    var train_num = X.rows
    var numVars = X.cols
    
    //参数配置， 并且广播配置
    var svmconfig = SVMConfig(numVars: Int,
                              gamma: Double,
//                            sum: Double,
                              C: Double,
                              eps: Double,
                              tolerance: Double,
                              train_num: Int)
                              
//    var weightconfig = WeightConfig(
//                              i: Int,
//                              j: Int,
//                              initW: Array[BDM[Double]],
//                              initA: Array[BDM[Double]],
//                              b: Double,
//                              error_cache: BDM[Double])
                              
    
    
    //初始化权重
    var svm_W = SVMO.InitialWeight(numVars)
    if (!((initW.length == 1 ) && (initW(0) == (BDM.zeros[Double](1, 1))))){
      for (i <- 0 to initW.length - 1) {
        svm_W(i) = initW(i)
      }
    }
    
    var svm_A = SVMO.InitialAlpha(svmconfig.train_num)
    if (!((initA.length == 1 ) && (initA(0) == (BDM.zeros[Double](1, 1))))){
      for (i <- 0 to initA.length - 1) {
        svm_A(i) = initA(i)
      }
    }
    
    var svm_b = 0.0
    var errorcache = BDM.zeros[Double](train_num, 1)
    
    
    //广播参数
    val config = sc.broadcast(svmconfig)
    for(i <- 0 until config.value.train_num) {
    initStartTime = System.currentTimeMillis()
//尝试将变量打包 
//    val Weightconfig = sc.broadcast(weightconfig)
    
    //权重
    val alpha = sc.broadcast(svm_A(0))
    val W = sc.broadcast(svm_W(0))
    val b = sc.broadcast(svm_b)
    val error_cache = sc.broadcast(errorcache)
    
    //循环有问题
    val j = SVMO.examineExample(train_d, i, config, error_cache, alpha, b)
    
    val a_b = SVMO.takeStep(train_d, error_cache, i, j, config, alpha, b)
    
     svm_A(0) = a_b._1
     svm_b = a_b._2
     
     
    
    //val train_smo = SVMO.SVM(train_d, config, alpha, b)
    
    
    
    
//    //smo
//    val train_smo = SVMO
    //TODO
    
    initEndTime = System.currentTimeMillis()
    }
  
    //返回模型，即参数Weight,alpha,b
    new SvmModel(svmconfig, svm_A(0), svm_b)
  }
    
    
}

object SVMO extends Serializable {
  
  /*初始化权重
   * 初始化为一个很小的、接近0的随机值
   */
  def InitialWeight(size: Int): Array[BDM[Double]] = {
    val  svm_W = ArrayBuffer[BDM[Double]]()
    val d1 = BDM.rand(size, 1)
    d1 -= 0.5
    val f1 = 2*4*sqrt(6.0/(size))
    val d2 = d1*f1
    svm_W += d2
    svm_W.toArray
  }
  
  //初始化Alpha
  def InitialAlpha(size: Int): Array[BDM[Double]] = {
    val  svm_A = ArrayBuffer[BDM[Double]]()
    val d1 = BDM.rand(size, 1)
    d1 -= 0.5
    val f1 = 2*4*sqrt(6.0/(size))
    val d2 = d1*f1
    svm_A += d2
    svm_A.toArray
  }
  
   //目标函数
     def dot(Xi: BDV[Double],Xj:BDV[Double]): Double = Bsum(Xi:*Xi -Xi:*Xj -Xi:*Xj + Xj:*Xj)

   //计算eta   
    def eta(Xi: BDV[Double],
            Xj:BDV[Double],
            config: org.apache.spark.broadcast.Broadcast[SVMConfig]): Double ={
            val guass: Double = {
            val K11 = exp(-config.value.gamma * SVMO.dot(Xi, Xi))
            val K12 = exp(-config.value.gamma * SVMO.dot(Xi, Xj))
            val K22 = exp(-config.value.gamma * SVMO.dot(Xj, Xj))
            2*K12 - K11 - K22
            } 
            return guass
            }
     
	 	//目标函数  learned_func
	   def obj(X: BDM[Double],
	       Y: BDM[Double],
	       j:Int,
	       config: org.apache.spark.broadcast.Broadcast[SVMConfig],
	       alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],    
	       b: Double): Double = {//org.apache.spark.broadcast.Broadcast[Double]
        {for (i <- 0 until config.value.train_num) 
          
          yield alpha.value(i,0)*Y(i,0)*SVMO.dot(X(i,::).inner, X(j,::).inner)}.sum-b//.value//Q？ 
      }   
	   
	 //计算E(i)
     def Ei(X: BDM[Double],
         Y: BDM[Double],
         alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
         i:Int,
         b: Double,//org.apache.spark.broadcast.Broadcast[Double],
         config: org.apache.spark.broadcast.Broadcast[SVMConfig]): Double = {
     val fx = SVMO.obj(X, Y, i, config, alpha, b)
     fx-Y(i,0)
       }
     
           //计算L、H
  
      def L(Yi: Double,
            Yj: Double,
            alpha1: Double,
            alpha2: Double,
            config: org.apache.spark.broadcast.Broadcast[SVMConfig]):Double = {
        val L = {
          if (Yi!= Yj) {
          if (0 >= alpha2-alpha1) 0 else alpha2-alpha1
        }
        else if (0 >= alpha2 + alpha1 - config.value.C) 0 else alpha2 + alpha1 - config.value.C
        }
        return L
      }
       
      def H(Yi: Double,
            Yj: Double,
            alpha1: Double,
            alpha2: Double,
            config: org.apache.spark.broadcast.Broadcast[SVMConfig]):Double = {        
        val H = {
          if (Yi!= Yj) {
        
          if (config.value.C >= config.value.C + alpha2-alpha1) config.value.C + alpha2-alpha1 else config.value.C
        }
        else if (config.value.C >= alpha2 + alpha1 ) alpha2 + alpha1 else config.value.C
        }
        return H
        }
    
      
          
          
  //更新 a1,a2
    def renew_alpha_b(X: BDM[Double],
                     Y: BDM[Double],
                     i:Int,
                     j:Int,
                     Ei: Double,
                     Ej: Double,               
                     alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
                     b: Double,//org.apache.spark.broadcast.Broadcast[Double],
                     config: org.apache.spark.broadcast.Broadcast[SVMConfig]):(Double,Double,Double) = {
        val alpha1 = alpha.value(i,0)
        val alpha2 = alpha.value(j,0)
        val Yi = Y(i,0)
        val Yj = Y(j,0)
        val Xi = X(i,::).inner
        val Xj = X(j,::).inner
        println("Xilen" + Xi.length + "Xjlen" + Xj.length)
        var alpha2newclipped =  0.0
        val eta =  SVMO.eta(Xi, Xj, config)
        val L =  SVMO.L(Yi, Yj, alpha1, alpha2, config)
        val H = SVMO.H(Yi, Yj, alpha1, alpha2, config)
        val alpha2positive = alpha2 - Y(i,0) * (Ei-Ej)/eta
        val s = Y(i,0)*Y(j,0)
        val K11 = SVMO.dot(Xi, Xi)
        val K12 = SVMO.dot(Xi, Xj)
        val K22 = SVMO.dot(Xj, Xj)
        if (eta>0.001) {
          val alpha2new = alpha2positive
          if (alpha2new >= H) {val alpha2newclipped = H }
          else if (alpha2new <= L) { val alpha2newclipped = L}
          else {val alpha2newclipped = alpha2new}
        }
        else {
          val func1 = Y(i,0)*(Ei+ b) - alpha1*K11 - s*alpha2*K12
          val func2 = Y(j,0)*(Ej+ b) - s*alpha1*K12 - alpha2*K22
          val L1 = alpha1 + s*(alpha2-L)
          val H1 = alpha1 + s*(alpha2-H)
          val Lobj = L1* func1 + L*func2 + 1/2 * L1*L1*K11 + 1/2*L1*L1*K22 + s*L*L1*K12
          val Hobj = H1* func1 + H*func2 + 1/2 * H1*H1*K11 + 1/2*H1*H1*K22 + s*H*L1*K12
          if (Lobj > Hobj +config.value.eps) {alpha2newclipped = L }
          else if (Lobj < Hobj - config.value.eps){alpha2newclipped = H}
          else {alpha2newclipped = alpha2}
        } 
        
        val alpha1new = alpha1 + s*(alpha2 - alpha2newclipped)  
        
        
             //计算b
      
      var bnew: Double = {
      if (alpha1new > 0 && alpha1new < config.value.C)
      {
       Ei + Y(i,0)*(alpha1new - alpha1)* K11 + Y(j,0)*(alpha2newclipped - alpha2)* K12 + b
      }
      else {
        if (alpha2newclipped > 0 && alpha2newclipped < config.value.C) {
        Ej + Y(i,0)*(alpha1new - alpha1)* K12 + Y(j,0)*(alpha2newclipped - alpha2)* K22 + b
        }
        else {
          var b1 = Ei + Y(i,0)*(alpha1new - alpha1)* K11 + Y(j,0)*(alpha2newclipped - alpha2)* K12 + b
          var b2 = Ej + Y(i,0)*(alpha1new - alpha1)* K12 + Y(j,0)*(alpha2newclipped - alpha2)* K22 + b
         (b1+ b2)/2
        }
        }
        }
        
      var bb = bnew
      
      return (alpha1new,alpha2newclipped,bb)
    }
    
//    def SVM(train_d: RDD[(BDM[Double],BDM[Double])],
//             config: org.apache.spark.broadcast.Broadcast[SVMConfig],
//             svm_W: org.apache.spark.broadcast.Broadcast[BDM[Double]],
//             svm_A: org.apache.spark.broadcast.Broadcast[BDM[Double]],
//             b: org.apache.spark.broadcast.Broadcast[Double]): (BDM[Double],BDM[Double],BDM[Double],Double) = {
//       var X = train_d.map(f => f._2).collect().apply(0)
//       var Y = train_d.map(f => f._1).collect().apply(0)
//       var alpha = svm_A.value
//       var train_num = X.rows
//       var error_cache = Array[Double](train_num)
//       for (i <- 0 until train_num) {
//         SVMO.examineExample(X, Y, alpha, error_cache, i, config, svm_W, svm_A, train_num, b)
//       }
//       //TODO
//       (Y,svm_W.value,svm_A.value,b.value)
//    }
             
     
   //优化两个乘子，成功返回1，否则0
    def takeStep(train_d: RDD[(BDM[Double],BDM[Double])],
        error_cache: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        i: Int,
        j:Int,       
        config: org.apache.spark.broadcast.Broadcast[SVMConfig],
//        svm_W: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        b: org.apache.spark.broadcast.Broadcast[Double]) : (BDM[Double],Double,BDM[Double]) = {//
      //将数据取出
        val X = train_d.map(f => f._2).collect().apply(0)
        val Y = train_d.map(f => f._1).collect().apply(0)
//        val train_num = X.rows
        var Ei = 0.0
        var Ej = 0.0
//        if (i == j) return 0  //不会优化两个相同算子        
        val alpha1 = alpha.value(i,0)
        val alpha2 = alpha.value(j,0)
//        var Xi = X(i,::).inner
//        var Xj = X(j,::).inner
        val Yi = Y(i,0)
        val Yj = Y(j,0)
        
        
        if (alpha1 > 0 && alpha1< config.value.C ) Ei = error_cache.value(i,0)
        else Ei = SVMO.obj(X, Y, i, config, alpha, b.value) - Yi
        
        if (alpha2 > 0 && alpha2 < config.value.C ) Ej = error_cache.value(j,0)
        else Ej = SVMO.obj(X, Y, j, config, alpha, b.value) - Yj
        
        //计算乘子的上下界
//        var L = SVMO.L(Yi, Yj, alpha1, alpha2, config.value.C )
//        var H = SVMO.H(Yi, Yj, alpha1, alpha2, config.value.C )
//        var eta = SVMO.eta(Xi, Xj, config.value.gamma)
        
        var a_b = SVMO.renew_alpha_b(X, Y, i, j, Ei, Ej, alpha, b.value, config)
        
        var delta_b = a_b._3 - b.value
        var bnew = a_b._3 
        //更新 error_cache
        var errorcache = SVMO.error_cache(X, Y, i, j, error_cache, alpha, config,a_b._1, a_b._2, delta_b)
//        var W = SVMO.SVM_W(X, svm_W.value, alpha.value, i, j, alpha1new, alpha2new, Yi, Yj, train_num)
        
        var alph = BDM.zeros[Double](config.value.train_num,1)
        for(k <- 0 until config.value.train_num){
          alph(k,0) = alpha.value(k,0)
        }
        
        alph(i,0) = a_b._1
        alph(j,0) = a_b._2
        
        (alph,bnew,errorcache)
        
        }
    
    //存放non-bound样本误差
    def error_cache(X: BDM[Double],
        Y: BDM[Double],
        i: Int,
        j: Int,
        error_cache: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        config: org.apache.spark.broadcast.Broadcast[SVMConfig],
//        train_num: Int,
        alpha1new: Double,
        alpha2new: Double,
//        b: org.apache.spark.broadcast.Broadcast[Double],
        delta_b: Double): BDM[Double] ={   
      val K12 = exp(-config.value.gamma * SVMO.dot(X(i,::).inner, X(j,::).inner))
      val K22 = exp(-config.value.gamma * SVMO.dot(X(j,::).inner, X(j,::).inner))
      var t1 = Y(i,0)*(alpha1new - alpha.value(i,0))
      var t2 = Y(j,0)*(alpha2new - alpha.value(j,0))
      val errorcache = BDM.zeros[Double](config.value.train_num,1)
      for(k <- 0 until config.value.train_num) {
        if (alpha.value(k,0)>0 && alpha.value(k,0) < config.value.C) {
          val K_ik = exp(-config.value.gamma * SVMO.dot(X(i,::).inner, X(k,::).inner))
          val K_jk = exp(-config.value.gamma * SVMO.dot(X(j,::).inner, X(k,::).inner))
          errorcache(k,0) += t1*K_ik+ t2*K_jk - delta_b
        }
      }
      errorcache(i,0) = 0
      errorcache(j,0) = 0
      return errorcache
      
    }
    
    //1: 在non_bound乘子中寻找maximunfabs(E1-E2)的样本
    def examineFirstChoice(
        error_cache: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        i: Int, 
        Ei: Double,
        config: org.apache.spark.broadcast.Broadcast[SVMConfig],
//        svm_W: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
//        train_num: Int,
        b:org.apache.spark.broadcast.Broadcast[Double]): Int = {
      var j = 0
      for (  k <- 0 until config.value.train_num){
        var tmax,temp= 0.0
        
        if (alpha.value(k,0) > 0 && alpha.value(k,0) < config.value.C){
          var Ej = error_cache.value(k,0)
          temp = abs(Ei-Ej)
          if (temp > tmax){
            tmax = temp
            j = k
          }
        }
      }
      
//      if (j >= 0 && SVMO.takeStep(train_d, alpha, error_cache, train_num, i, j, config, svm_W, svm_A, b)==1) return 1
      if (i==j) return 0
      else return j 
    }
    
    //2:如果上面没取得进展，那么从随机位置查找non_boudary样本
    def examineNonBound(
        i: Int,
        config: org.apache.spark.broadcast.Broadcast[SVMConfig],
//        svm_W: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
//        train_num: Int,
        b:org.apache.spark.broadcast.Broadcast[Double]): Int = {
      var j= 0
      var rand = new Random()
      var k0 = rand.nextInt(config.value.train_num)
      var kk = for (k  <- 0 until config.value.train_num){
        j = (k+k0)% config.value.train_num
        if ((alpha.value(j,0)> config.value.eps && alpha.value(j,0)< config.value.C )&& i !=j) return j
        //TODO
         //希望有break的功能
      }
      return 0
    }
    
    //3: 如果上面也失败，则从随机位置查找整个样本，（bound样本）
    def examineBound(
        i: Int,
        config: org.apache.spark.broadcast.Broadcast[SVMConfig],
//        svm_W: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
//        train_num: Int,
        b:org.apache.spark.broadcast.Broadcast[Double]): Int = {
      var j  = 0
      var rand = new Random()
      val k0 = rand.nextInt(config.value.train_num)
      for(k <- 0 until config.value.train_num){
        j = (k+k0)% config.value.train_num
        if (i!=j) return j
        //TODO
        //希望有break的功能
      }
      return 0
    }
    
    /*假定第一个乘子a1，examineExample首先检查，如果他超出tolerace
     * 而违背KKT条件，那么他就成为第一个乘子
     * 然后，选找第二个乘子a2，通过调用takestep来优化这两个乘子
     
     */
    
    def examineExample(train_d: RDD[(BDM[Double],BDM[Double])],
        i: Int,
        config: org.apache.spark.broadcast.Broadcast[SVMConfig],
//      W: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        error_cache: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
        b:org.apache.spark.broadcast.Broadcast[Double]): Int = {
      val X = train_d.map(f => f._2).collect().apply(0)
      val Y = train_d.map(f => f._1).collect().apply(0)
//      val train_num = X.rows
      var Ei = 0.0
      
      if (alpha.value(i,0) > 0 && alpha.value(i,0) < config.value.C)  Ei = error_cache.value(i,0)
      else Ei = SVMO.obj(X, Y, i, config, alpha, b.value)
      var r1 = Y(i,0)* Ei
      var j = 0
      //违反KKT条件的判断
      if((r1 > config.value.tolerance && alpha.value(i,0)>0)||(r1 < - config.value.tolerance && alpha.value(i,0) < config.value.C))
      {
        /*
         * 使用三种方法选择第二个乘子
         * 1: 在non_bound乘子中寻找maximunfabs(E1-E2)的样本
         * 2:如果上面没取得进展，那么从随机位置查找non_boudary样本
         * 3: 如果上面也失败，则从随机位置查找整个样本，（bound样本）
         */
        j= SVMO.examineFirstChoice(error_cache, i, Ei, config, alpha, b)
        if (j == 0) j = SVMO.examineNonBound(i, config, alpha, b)
        if (j == 0) j = SVMO.examineBound(i, config, alpha, b)
      }
      return j
    }
    
//      SVM
      def SVM(train_d: RDD[(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double])],
          config: org.apache.spark.broadcast.Broadcast[SVMConfig],
          alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
          b:org.apache.spark.broadcast.Broadcast[Double]): RDD[(SVMLabel,BDM[Double])] = {
        val train_data1 = train_d.map{ f =>
          val Y = f._1
          val X = f._2
          val W = BDV.zeros[Double](config.value.numVars)
          for (k <- 0 until config.value.numVars) W(k) = {
          for(i <- 0 until config.value.train_num) yield  alpha.value(i,0)* Y(i,0)* X(i,k)}.sum
          println("Xilen" + X.cols + "Wlen" + W.length)
          val YY = BDM.zeros[Double](config.value.train_num, 1)
          
        
          for (i <- 0 until config.value.train_num) {
          YY(i, 0) = SVMO.dot(W, X(i,::).inner)+ b.value
          if (YY(i,0) > 0) YY(i,0) = 1 else YY(i,0) = 0}
          var count_err = 0
          for (i <- 0 until config.value.train_num){
           if (YY(i,0) == Y(i,0)) count_err += 1
           count_err         
          }
          
           val err_rate = count_err.toDouble / config.value.train_num.toDouble
           
          
         ( SVMLabel(Y, YY, err_rate ),f._2)
        }
        
          //计算出W，返回出 W，b，有可能是输出结果，错误率
    
        
        
        train_data1
        
          
        
        
        //TODO
      }
      
//   //更新W,Alpha
//    def SVM_W(X: BDM[Double],
//                svmW: BDM[Double],
//                alpha: BDM[Double],
//                i: Int,
//                j: Int,
//                alpha1new: Double,
//                alpha2new: Double,
//                Yi: Double,
//                Yj: Double,
//                train_num: Int): BDM[Double] = {
//      for ( k <- 0 until train_num) svmW(k,0) = svmW(k,0) + Yi*(alpha1new - alpha(i,0))*X(i,k) + Yj*(alpha2new - alpha(j,0))*X(j,k) 
//      return svmW
//    }
    
//    def error_rate(X:BDM[Double],
//        Y: BDM[Double],
//        alpha: org.apache.spark.broadcast.Broadcast[BDM[Double]],
//        i: Int, 
//        j: Int,
//        b:org.apache.spark.broadcast.Broadcast[Double],
//        E1: Double,
//        C : Double,
//        train_num: Int,
//        N: Int): Double = {
//      var ac = 0
//      var accuracy,tar = 0.0
//      println("--------------测试结果-----------")
//      for ( k <- train_num until N){
//        tar = SVMO.obj(X, Y, j, train_num, alpha, b)
//        if(tar>0 && Y(k,0) > 0 ||tar < 0 && Y(k,0)< 0 ) ac += 1       
//      }
//      accuracy
//      
//          
//    }
}
        
    
//        //计算乘子的上下限
//        var Low = 0.0
//        var Hig = 0.0
//        if (s== 1 ){
//          var ga = alpha1 + alpha2
//          if ( ga > C) {
//             Low = ga - C
//             Hig = C
//          }
//          else {
//             Low = 0
//             Hig = ga
//          }         
//          }
//        else {
//          var ga = alpha1 - alpha2 
//          if (ga > 0 ){
//             Low = 0
//             Hig = C - ga
//          }
//          else {
//             Low = - ga
//             Hig = C
//          }
//        }
//        
//        if ( Low == Hig) return 0

    
    //放入了takeStep中，后期考虑输出成模块    
//    //计算eta   
//    def eta(i: Int,
//            j: Int,
//            gamma: Double): Double ={
//            val guass: Double = {
//            val K11 = exp(-gamma * dot(i,i));
//            val K12 = exp(-gamma * dot(i,j));
//            val K22 = exp(-gamma * dot(j,j));
//            2*K12 - K11 - K22
//            }
//           return guass;     
//    }
    
//          if (eta < -0.001) {
//            val c = Y2* ( E2- E1)
//            var alpha2new = alpha2 + c/eta //计算新的alpha2
//            //调整a2，使其处于可行域
//            if (alpha2new < L) alpha2new = L
//            else if (alpha2new > H) alpha2new = H
//            else alpha2new = alpha2new
//          }
//          else { //分别从端点H，L求目标函数值Lobj，Hobj，然后设alpha2new为所求最大目标函数值
//            var Lobj = SVMO.L(Y, alpha, i, j, C)
//            var Hobj = SVMO.H(Y, alpha, i, j, C)
//            if (Lobj > Hobj + eps) alpha2new = L
//            else if (Hobj > Lobj + eps) alpha2new = H
//            else alpha2new = alpha2
//            
//          }
//        
//          if (fabs(alpha2new - alpha2)< eps) return 0
//          
//            alpha1new = alpha1 - s*(alpha2new - alpha2) //计算新的alpha1
//            if (a1 < 0) {
//              alpha2new += s*alpha1new
//              alpha1new = 0
//            }
//            else if ( alpha1new > C){
//              alpha2new += s*(alpha1new - C)
//              alpha1new = C
//            }            
//          
//        var bnew = SVMO.renew_alpha_b(X, Y, alpha, i, j, sum, gamma, b, C, eps, numVars)
//        delta_b = bnew - b
