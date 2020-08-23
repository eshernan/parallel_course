# Parallel computing course
This repositorory containt  the resources used on the course 2020-3 of parallel computing 

## prerequisites

 Install the gcc and g++ compiler a <br/>
```
$> # for ubuntu base 
$> sudo apt install -y gcc g++   <br/>
$> gcc --version ; g++ --version 
```


```
$># For Centos Base
$> yum groupinstall "Development Tools"
```

### test a base code 
Create a compile a test_pthreads.c file with the following content 


```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *print_message_function( void *ptr );

main()
{
     pthread_t thread1, thread2;
     char *message1 = "Thread 1";
     char *message2 = "Thread 2";
     int  iret1, iret2;

    /* Create independent threads each of which will execute function */

     iret1 = pthread_create( &thread1, NULL, print_message_function, (void*) message1);
     iret2 = pthread_create( &thread2, NULL, print_message_function, (void*) message2);

     /* Wait till threads are complete before main continues. Unless we  */
     /* wait we run the risk of executing an exit which will terminate   */
     /* the process and all threads before the threads have completed.   */

     pthread_join( thread1, NULL);
     pthread_join( thread2, NULL); 

     printf("Thread 1 returns: %d\n",iret1);
     printf("Thread 2 returns: %d\n",iret2);
     exit(0);
}

void *print_message_function( void *ptr )
{
     char *message;
     message = (char *) ptr;
     printf("%s \n", message);
}
```


<p> Compile and run the file <br/>

```console
> gcc test_pthreads.c -o test_pthreads
> ./test_pthreads
Thread 1 
Thread 2 
Thread 1 returns: 0
Thread 2 returns: 0
```
