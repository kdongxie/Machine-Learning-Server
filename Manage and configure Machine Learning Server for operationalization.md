

# Manage and configure Machine Learning Server for operationalization

## Machine Learning Server의 두가지 형태 

#### 1. Enterprise

A [enterprise configuration](https://docs.microsoft.com/en-us/machine-learning-server/operationalize/configure-machine-learning-server-enterprise) where multiple nodes are configured on multiple machines along with other enterprise features. This configuration can be scaled out or in by adding or removing nodes. Learn more about this setup in the [enterprise configuration](https://docs.microsoft.com/en-us/machine-learning-server/operationalize/configure-machine-learning-server-enterprise) article. For added security, you can [configure SSL](https://docs.microsoft.com/en-us/machine-learning-server/operationalize/configure-https) and authenticate against [Active Directory (LDAP) or Azure Active Directory](https://docs.microsoft.com/en-us/machine-learning-server/operationalize/configure-authentication) in this configuration.

![image](https://user-images.githubusercontent.com/46669551/54585983-f936a500-4a5e-11e9-9a28-ce344974521a.png)

#### 2. One-box

As the name suggests, a [one-box configuration](https://docs.microsoft.com/en-us/machine-learning-server/operationalize/configure-machine-learning-server-one-box) involves one web node and one compute node run on a single machine. Set-up is a breeze. This configuration is useful when you want to explore what it is to operationalize R and Python analytics using Machine Learning Server. It is perfect for testing, proof-of-concepts, and small-scale prototyping, but might not be appropriate for production usage. This configuration is covered in this article. Learn more in this [One-box configuration](https://docs.microsoft.com/en-us/machine-learning-server/operationalize/configure-machine-learning-server-one-box) article.

![image](https://user-images.githubusercontent.com/46669551/54584052-91ca2680-4a59-11e9-85df-1d4d89722553.png)



### 아래의 링크를 통해 설치를 실시

https://docs.microsoft.com/en-us/machine-learning-server/install/machine-learning-server-windows-install

### Download 를 진행하여  Server로 활용할 Machine에 설치한다.

![image](https://user-images.githubusercontent.com/46669551/54582553-832d4080-4a54-11e9-852c-db91a1ae6a65.png)



![image](https://user-images.githubusercontent.com/46669551/54582576-9b9d5b00-4a54-11e9-825d-31c2bcb9c2c2.png)

![image](https://user-images.githubusercontent.com/46669551/54582594-aeb02b00-4a54-11e9-80f2-513febc9dfd4.png)



![image](https://user-images.githubusercontent.com/46669551/54582675-f9ca3e00-4a54-11e9-9cd5-3a2218cf9655.png)



![image](https://user-images.githubusercontent.com/46669551/54582616-bf60a100-4a54-11e9-93f3-2c024f41f393.png)

![image](https://user-images.githubusercontent.com/46669551/54582639-d69f8e80-4a54-11e9-86ed-693f139d83ac.png)

![image](https://user-images.githubusercontent.com/46669551/54582746-3a29bc00-4a55-11e9-872f-e2ecc376bc5d.png)



### 원격으로 접속할 Client 서버에 R Client를 설치

# R Client 설치 

```
https://aka.ms/rclient/
```



### R Server가 설치된 Machine으로 이동하여 Server로써의 구성을 실시

# R Setting

```
Connect and validate
Machine Learning Server executes on demand as R Server or as a Python application. As a verification step, connect to each application and run a script or function.

For R

R Server runs as a background process, as Microsoft ML Server Engine in Task Manager. Server startup occurs when a client application like R Tools for Visual Studio or Rgui.exe connects to the server.

1. Go to C:\Program Files\Microsoft\ML Server\R_SERVER\bin\x64.
2. Double-click Rgui.exe to start the R Console application.
3. At the command line, type search() to show preloaded objects, including the RevoScaleR package.
4. Type print(Revo.version) to show the software version.
5. Type rxSummary(~., iris) to return summary statistics on the built-in iris sample 	dataset. The rxSummary function is from RevoScaleR.


```



# Python Setting

Python runs when you execute a .py script or run commands in a Python console window.

1. Go to C:\Program Files\Microsoft\ML Server\PYTHON_SERVER.
2. Double-click **Python.exe**.
3. At the command line, type `help()` to open interactive help.
4. Type `revoscalepy` at the help prompt to print the package contents.
5. Paste in the following revoscalepy script to return summary statistics from the built-in AirlineDemo demo data:

```python
import os
import revoscalepy 
sample_data_path = revoscalepy.RxOptions.get_option("sampleDataDir")
ds = revoscalepy.RxXdfData(os.path.join(sample_data_path, "AirlineDemoSmall.xdf"))
summary = revoscalepy.rx_summary("ArrDelay+DayOfWeek", ds)  
print(summary)
```

Output from the sample dataset should look similar to the following:


  ```python
Summary Statistics Results for: ArrDelay+DayOfWeek
File name: ... AirlineDemoSmall.xdf
Number of valid observations: 600000.0

        Name       Mean     StdDev   Min     Max  ValidObs  MissingObs
0  ArrDelay  11.317935  40.688536 -86.0  1490.0  582628.0     17372.0

Category Counts for DayOfWeek
Number of categories: 7

            Counts
DayOfWeek         
1          97975.0
2          77725.0
3          78875.0
4          81304.0
5          82987.0
6          86159.0
7          94975.0
  ```



# Azure Setting

open admin command prompt window

Enter the following command to check availability of the CLI: `az ml admin --help`. If you receive the following error: `az: error argument _command_package: invalid choice: ml`, follow the instructions to re-add the extension to the CLI.

##### Enable web service deployment and remote connections

1. Open an Administrator command prompt.
2. Enter the following command to configure the server: `az ml admin bootstrap`

![image](https://user-images.githubusercontent.com/46669551/54584189-0d2bd800-4a5a-11e9-84e3-1996136e4896.png)





# R Studio for Client  

#### R studio를 실행시켜 Global Option을 확인하면, Default값이 R_Server에 위치함

![image](https://user-images.githubusercontent.com/46669551/54584375-8297a880-4a5a-11e9-8b13-16035aff6bea.png)

### Command를 이용하여 R Server에 연결

#### R-Studio에 아래의 Command를 실행하여 원격으로 접속이 가능한지 확인한다. 

```R
remoteLogin(deployr_endpoint = "http://172.16.0.102:12800",session = TRUE, diff = TRUE, commandline = TRUE, username = "admin", password = "Pa$$w0rd")
```



