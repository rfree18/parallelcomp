#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/*
Conditions for convergence (diagonal dominance):
1. diagonal element >= sum of all other elements of the row
2. At least one diagonal element > sum of all other elements of the row
*/
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;

  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);

    for(j = 0; j < num; j++)
    if( j != i)
    sum += fabs(a[i][j]);

    if( aii < sum)
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }

    if(aii > sum)
    bigger++;

  }

  if( !bigger )
  {
    printf("The matrix will not converge\n");
    exit(1);
  }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
* a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
* x[] will contain the initial values of x
* b[] will contain the constants (i.e. the right-hand-side of the equations
* num will have number of variables
* err will have the absolute error that you need to reach
*/
void get_input(char filename[])
{
  FILE * fp;
  int i,j;

  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

  fscanf(fp,"%d ",&num);
  fscanf(fp,"%f ",&err);

  /* Now, time to allocate the matrices and vectors */
  a = (float**)malloc(num * sizeof(float*));
  if( !a)
  {
    printf("Cannot allocate a!\n");
    exit(1);
  }

  for(i = 0; i < num; i++)
  {
    a[i] = (float *)malloc(num * sizeof(float));
    if( !a[i])
    {
      printf("Cannot allocate a[%d]!\n",i);
      exit(1);
    }
  }

  x = (float *) malloc(num * sizeof(float));
  if( !x)
  {
    printf("Cannot allocate x!\n");
    exit(1);
  }


  b = (float *) malloc(num * sizeof(float));
  if( !b)
  {
    printf("Cannot allocate b!\n");
    exit(1);
  }

  /* Now .. Filling the blanks */

  /* The initial values of Xs */
  for(i = 0; i < num; i++)
  fscanf(fp,"%f ", &x[i]);

  for(i = 0; i < num; i++)
  {
    for(j = 0; j < num; j++)
    fscanf(fp,"%f ",&a[i][j]);

    /* reading the b element */
    fscanf(fp,"%f ",&b[i]);
  }

  fclose(fp);

}


/************************************************************/

int check_error(float *new_val, int size) {
  int i = 0;

  for(i = 0; i < num; i++) {
    float curr_error = fabs((new_val[i] - x[i]) / new_val[i]);

    if(curr_error > err) {
      // Error is greater than threshold
      return 1;
    }
  }

  // All errors are below threshold
  return 0;
}


int main(int argc, char *argv[])
{

  int i;
  int nit = 0; /* number of iterations */


  if( argc != 2)
  {
    printf("Usage: gsref filename\n");
    exit(1);
  }

  /* Read the input file and fill the global data structure above */
  get_input(argv[1]);

  /* Check for convergence condition */
  /* This function will exit the program if the coffeicient will never converge to
  * the needed absolute error.
  * This is not expected to happen for this programming assignment.
  */
  check_matrix();

  //Initialize MPI...
  int comm_sz;
  int my_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int num_eq = num / comm_sz;

  int *proc_disp = malloc(comm_sz * sizeof(int));
  proc_disp[0] = 0;

  // for(i = 0; i < comm_sz; i++) {
  //   if(i < rem) {
  //     eq_count[i] = num_eq + 1;
  //   } else {
  //     eq_count[i] = num_eq;
  //   }
  //
  //   if(i != 0) {
  //     proc_disp[i] = proc_disp[i-1] + eq_count[i-1];
  //   }
  // }

  // int local_size = eq_count[my_rank];

  float *local_b = malloc(num_eq * sizeof(float));
  float *local_x = malloc(num_eq * sizeof(float));
  float *new_x = malloc(num * sizeof(float));

  for(i = 0; i < num; i++) {
    new_x[i] = x[i];
  }

  // Distribute b and x values to processes depending on assignments
  // NOTE: a is not sent due to complications of double pointers
  // MPI_Scatterv(b, eq_count, proc_disp, MPI_FLOAT, local_b, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Scatterv(x, eq_count, proc_disp, MPI_FLOAT, local_x, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter(b, num_eq, MPI_FLOAT, local_b, num_eq, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter(x, num_eq, MPI_FLOAT, local_x, num_eq, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int a_base = 0;

  for(i = 0; i < my_rank; i++) {
    // Get starting point for needed a values
    a_base += num_eq;
  }

  do {
    nit++;
    // Save old values to check error
    for(i = 0; i < num; i++){
          x[i] = new_x[i];
      }
    for(i = 0; i < num_eq; i++) {
      int j;
      int start_index = a_base + i;

      local_x[i] = local_b[i];

      for(j = 0; j < num; j++) {
        if(j != start_index) {
          local_x[i] = local_x[i] - a[start_index][j] * x[j];
        }
      }
      local_x[i] = local_x[i] / a[start_index][start_index];
    }

    // Adjust start address to only send assigned values
    // MPI_Allgatherv(local_x, eq_count[my_rank], MPI_FLOAT, new_x, eq_count, proc_disp,
    //   MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(local_x, num_eq, MPI_FLOAT, new_x, num_eq, MPI_FLOAT, MPI_COMM_WORLD);
  } while(check_error(new_x, num));

  // printf("Process: %d\n", my_rank);

  /* Writing to the stdout */
  /* Keep that same format */
  if (my_rank == 0){
    for( i = 0; i < num; i++)
    printf("%f\n",new_x[i]);

    printf("total number of iterations: %d\n", nit);
    free(a);
    free(b);
    free(x);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  free(local_x);
  free(local_b);
  free(new_x);
  free(proc_disp);

  MPI_Finalize();

  exit(0);

}
