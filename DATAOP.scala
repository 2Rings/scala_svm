//package SVM
//
//import scala.collection.mutable.HashMap
//import org.apache.spark.SparkConf
//import org.apache.spark.SparkContext
//import org.apache.spark.mllib.evaluation.MulticlassMetrics
//import org.apache.spark.mllib.linalg.Vector
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.rdd.RDD
//
//import SVMLabel._
//
//import breeze.linalg.{ DenseMatrix => BDM }
//import breeze.linalg.{ DenseVector => BDV }
//import breeze.linalg.{ Matrix => BM }
//import breeze.linalg.{ Vector => BV }
//import breeze.linalg.{ axpy => brzAxpy }
//import scala.collection.mutable.ArrayBuffer
//
//case class PredictSVMLabel(label: BDM[Double],err: Double) extends Serializable
//
//class SVMModel(
//  val config: SVMConfig,
//  val alpha: BDM[Double],
//  val initb: Double) extends Serializable{ 
//  
//  def predict(data: RDD[(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double])]): PredictSVMLabel = {
//    val sc = data.sparkContext
////    val svm_W = sc.broadcast(weights)
//    val svm_A = sc.broadcast(alpha)
//    val b = sc.broadcast(initb)
//    val svmconfig = sc.broadcast(config)
//    val train_svm = SVMO.SVM(data, svmconfig, svm_A, b)
//    val predict = train_svm.map{ f => 
//      val label = f._1.label
//      val err = f._1.err
//      
//      PredictSVMLabel(label,err)
//    }
//  }
//    
//  def classification(data: RDD[(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double])],
//      test:RDD[breeze.linalg.DenseMatrix[Double]]): RDD[BDM[Double]] = {
//    val sc = data.sparkContext
////    val svm_W = sc.broadcast(weights)
//    val svm_A = sc.broadcast(alpha)
//    val b = sc.broadcast(initb)
//    val svmconfig = sc.broadcast(config)
//    val testX = test.collect().apply(0)
//    val test_d = data.map{ f =>
//      val label = f._1
//      val features = f._2
//      val forcast = BDM.zeros[Double](label.rows,1)
//      for (i <- 0 until label.rows){
//        forcast(i,0) = {for (k <- 0 until label.rows) yield svm_A.value(k,0)*SVMO.dot(label(k,::).inner, testX(i,::).inner)}.sum
//      }
//      forcast
//    }
//    test_d
//  }
//    //将数据取出
////    val X = data.map(f => f._2).collect().apply(0)
////    val Y = data.map(f => f._1).collect().apply(0)
//    
////    val 
////    val YY = BDM.zeros[Double](svmconfig.value.train_num,1)
////    for (i <- 0 until svmconfig.value.train_num) {
////      YY(i,0) = svm_A.value(i,0) * SVMO.dot(X(i,::).inner, X(j,::).inner)
////    }
////    
////    predict
//    
////    def error_rate(data: RDD[(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double])]): Double ={
//      
////    }
//  }
//  
//  
//  
//}
//object DataOp {
//  
//  def operation(sc: SparkContext, 
//                train_data_source: String, 
//                test_data_source: String, 
//                name_source: String, 
//                target_name: String, 
//                data_type_source: String){
//    val train_tmp = sc.textFile(train_data_source).collect.take(1000)  
//    val train = sc.parallelize(train_tmp.drop(1)) //去掉第一行，即变量名
//    val test_tmp = sc.textFile(test_data_source).collect.take(1000)
//    val test = sc.parallelize(test_tmp.drop(1))
//    val name = sc.textFile(name_source).collect
//    val target = target_name 
//    val targetIndex = name.indexOf(target)  //找出目标变量所在列
//    val data_type = sc.textFile(data_type_source).collect
//    val types = data_type.map{line => val value = line.split(',').map(_.toInt); value(0)}
//    val algo = if (types(targetIndex) == 1) "Classification" else "Regression"
//      
//    val trainData = dataToLabeledPoint(train, target, name) //调用dataToLabeledPoint,处理数据
//    val testData = dataToLabeledPoint(test, target, name)
//    
//    val train_d = norm(trainData) //归一化
//    val test_d = norm(testData)
//    
//    val model = modelPack(train_d)
//    Evalue(model, trainData, testData, algo)
//    
//    
//  }
//    def dataToLabeledPoint(data:RDD[String], target: String, name: Array[String]): RDD[LabeledPoint] = {
//      val data_tmp1 = data
//      val targetIndex = name.indexOf(target)
//      val colLength = name.length
//      
//      //将数据设为LabeledPoint格式
//      def getLabeledPoint(targetIndex: Int, colLength: Int): RDD[LabeledPoint] = {
//        val data_tmp = data.map{ line =>
//          val values = line.split(',').map(_.toDouble)
//          val label = values(targetIndex)
//          val right = values.drop(targetIndex)
//          val left = values.dropRight(colLength-targetIndex+1)
//          val feature = left ++ right //去掉目标函数所在列后整合
//          val featureVector = Vectors.dense(feature)
//          LabeledPoint(label,featureVector)  //返回目标变量和自变量
//        }
//        
//        data_tmp
//      }
//      val data_ = getLabeledPoint(targetIndex, colLength)
//      data_
//    }
//    
//    def norm(data: RDD[LabeledPoint]): RDD[(breeze.linalg.DenseMatrix[Double], breeze.linalg.DenseMatrix[Double])] ={
//      val features = data.map(value => BDM(value.features.toArray)) //转成Array
//      val numExamples = features.count()
//      val normMax = features.map(_.max).max  //计算最值 
//      val normMin = features.map(_.min).min
//      
//      val vars = data.map(value => value.label) //计算target类别数目
//      val varDistinct = vars.distinct         //区分
//      val length = varDistinct.count.toInt
//      
//      val data_ = data.map{ f =>
//        val label = f.label
//        val labeltoOneHot = new Array[Double](length)
//        for (i <- 0 until length){
//          labeltoOneHot(i) = if (label ==1 ) 1.0 else 0.0
//        }
//        
//        //归一化公式 (x-min)/(max-min)
//        
//        val samp = BDM( f.features.toArray)
//        val sampNorm1 = samp - BDM((BDV.ones[Double](samp.size)*normMin).toArray)
//        val sampNorm2 = sampNorm1 :/ BDM((BDV.ones[Double](samp.size)*normMax - normMin).toArray)
//        
//        (BDM(labeltoOneHot),sampNorm2)
//      }
//      return data_
//    
//    }
//    
//    def modelPack[T](train_d: RDD[(breeze.linalg.DenseMatrix[Double],
//        breeze.linalg.DenseMatrix[Double])]): SVMModel = 
//        {
//      val numVars = train_d.first._2.size
//      val numExamples = train_d.count
//      
//      val numClasses = train_d.first._1.size
//      val opts = Array(100,20,0.0)   //一些步长，迭代次数的选项
//      val model = new SVMO().
//      setC(0.0).setEps(0.1).setGamma(0.1).SVMtrain(train_d)
//      return model
//      }
//    
//    //将哑变量换成类别输出 //TO be renew
//  def decodeOneHot(data: RDD[breeze.linalg.DenseMatrix[Double]]): RDD[Double] ={
//    val numVars = data.first.size
//    
//    val label_ = data.map{ value =>
//      val label = new Array[Double](1)
//      //TODO
//      for (i<- 0 until numVars){
//        val predsToClassiedValues = if (value(0,i) >= 0.5) 1.0 else 0.0
//        if (  predsToClassiedValues == 1.0) {label(0) = i}        
//      }
//      label(0)
//    }
//    label_
//  }
//  
//    def main(args: Array[String]): Unit = {
//    val spark_conf = new SparkConf().setMaster("local").setAppName("model");
//    val spark_context = new SparkContext(spark_conf)
//    val train_data_source = "D:/dmpro/data/DM_data/2/test/train_data.csv" //训练集输入
//    val test_data_source = "D:/dmpro/data/DM_data/2/test/test_data.csv" //测试集读入
//    val name_source = "D:/dmpro/data/DM_data/2/test/name.csv" //变量名读入
//    val target_name = "D:/dmpro/data/DM_data/2/test/target.csv" // 目标
//    val data_type_source = "D:/dmpro/data/DM_data/2/test/type.csv" //类型
//    
//    //调用operation， 对数据进行处理
//    operation(spark_context, train_data_source, test_data_source , name_source, target_name, data_type_source)
//  }
//  
//  //准确度
////  def getMetrics[T](model: SVMModel,
////      data: RDD[(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double])]): MulticlassMetrics = {
////    val oneHotPreds = model.predict(data).map( f => f.predict_label)
////    val oneHotLabel = data.map( f => f._1)
////    
////    val predictionsAndLabels = decodeOneHot(oneHotPreds).zip(decodeOneHot(oneHotLabel))
////    return new MulticlassMetrics(predictionsAndLabels)
////  }
////  
////  def classProbabilities(data:RDD[(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double])]): Array[Double] ={
////    val label = decodeOneHot(data.map(f=> f._1))
////    // Count (category,count) in data
////    val countsByCategory = label.countByValue()
////    // order counts by category and extract counts
////    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
////    counts.map(_.toDouble / counts.sum)
////    
////  }
////    
////  def Evalue[T](model: SVMModel, 
////      trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint],
////      algo: String): HashMap[String, Double]=
////  {  
////    val train_d = norm(trainData)
////    val test_d = norm(testData)
////    
////    if (algo == "Classification")
////    {
////      
////      val trainPriorProbabilities = classProbabilities(train_d)
////      val testPriorProbabilities = classProbabilities(test_d)
////      val random_precision = trainPriorProbabilities.zip(testPriorProbabilities).map {case (trainProb, testProb) => trainProb * testProb}.sum
////
////      val trainPrecision = getMetrics(model, train_d).precision
////      val testPrecision = getMetrics(model, train_d).precision
////      
////      println("trainPrecision: " + trainPrecision + "\n" +
////      "testPrecision: " + testPrecision + "\n" +
////      "random_precision: " + random_precision)
////      
////      val precision = HashMap[String,Double]() 
////      precision.put("trainPrecision", trainPrecision)  
////      precision.put("testPrecision",testPrecision)
////      precision.put("random_precision",random_precision)
////      
////      return precision
////      
////    } else
////    {
////      val train_d_max = trainData.map{f => val a1 = Array(f.label); val a2 = f.features.toArray; BDM(a1 ++ a2).max}.max
////      val train_d_min = trainData.map{f => val a1 = Array(f.label); val a2 = f.features.toArray; BDM(a1 ++ a2).max}.min
////      val train_dist = train_d_max - train_d_min
////      val test_d_max = trainData.map{f => val a1 = Array(f.label); val a2 = f.features.toArray; BDM(a1 ++ a2).max}.max
////      val test_d_min = trainData.map{f => val a1 = Array(f.label); val a2 = f.features.toArray; BDM(a1 ++ a2).max}.min
////      val test_dist = test_d_max - test_d_min
////      
////      val trainNNforecast = model.predict(train_d)
////      val testNNforecast = model.predict(test_d)
////    
////      val trainLabelAndPreds = trainNNforecast.map{f => 
////          val label = f.label.data(0) * train_dist + train_d_min
////          val preds = f.predict_label.data(0) * train_dist + train_d_min 
////          (label , preds)}
////    
////      val testLabelAndPreds = testNNforecast.map{f => 
////        val label = f.label.data(0) * train_dist + train_d_min
////        val preds = f.predict_label.data(0) * train_dist + train_d_min 
////        (label , preds)}
////      //evaluate test error
////      val trainLoss = trainLabelAndPreds.map {
////          case (l, p) =>
////          val err = l - p
////          err * err
////        }.reduce(_ + _)
////    
////      val testLoss = testLabelAndPreds.map {
////          case (l, p) =>
////          val err = l - p
////          err * err
////        }.reduce(_ + _)
////    
////      val trainRmse = math.sqrt(trainLoss / train_d.count)
////      val testRmse = math.sqrt(testLoss / test_d.count)
////    
////      println(s"Train RMSE = ${trainRmse}")
////      println(s"Test RMSE = ${testRmse}")
////        
////      val Rmse = HashMap[String,Double]()
////      Rmse.put("trainRmse", trainRmse)
////      Rmse.put("testRmse", testRmse)
////
////      return Rmse
//// 
////    }
////
////  }
//    
//
//  
//  
//  
//}
//
