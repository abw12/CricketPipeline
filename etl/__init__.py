from pyspark.sql import SparkSession

def createSparkSession():
    # create a spark session
    return (
        SparkSession.builder
        .appName("Cricket Pipeline") # type: ignore
        .config("inferSchema","true")
        .config("spark.master","local")
        .getOrCreate()
    )

def main():
    # fetch the spark session
    session: SparkSession = createSparkSession()
    match1_DF = session.read.json("data//raw//ipl//335982.json")
    match1_DF.show()
    
if __name__ == "__main__":
    main()
