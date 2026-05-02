/*
 * On src:   m(0:9,0:9) 
 *           send section m(2:6,4:9)
 *
 * On dest   m(0:11,0:7)
 *           recv section m(7:11,2:7)
 *
 * NOTE: this code emulates Fortran style indexing
 */
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define IND(i,j)  j*mx+i

int main( int argc, char *argv[] )
{
  //int m;
  int mx, my, nx, ny;
  int ierr, rank, size;
  int src, dest;
  MPI_Status status;
  int i,j;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 8;    
  mx = n;         //number of columns
  my = n / size;  //k rows per process
    
  //######################################
  // Task-1(c) for 2D instead of 1D
  //######################################
  int m[my][mx];
  //Allocate local matrix (k x n)
  //m = (int *) malloc(mx*my*sizeof(int));

  //Initialize matrix values
  for (int j = 0; j < my; j++){
    for (int i = 0; i < mx; i++){
      m[j][i] = rank * 100 + j * mx + i;
    }
  }
  //Print source array values with process rank
  printf("Process %d Source array:\n", rank);
  for(j=0; j<my; j++) {
    for(i=0; i<mx; i++){
	    //m[IND(i,j)] = IND(i,j);
	    printf("  %3d", m[j][i]);
    }
    printf("\n");
  }
  printf("\n");

  dest = (rank + 1) % size;       //Modify dest means right neighbour
  src = (rank -1 + size) % size;   //Modify src means left neighbour
 
  //##########################################
  // Task-1(a)
  //##########################################
  MPI_Datatype section_row;
  MPI_Type_contiguous( 2 * mx, MPI_INT , &section_row);
  MPI_Type_commit(&section_row);
    
  MPI_Sendrecv(
    &m[my - 2][0], 1, section_row, dest, 0,
    &m[0][0], 1, section_row, src, 0,
    MPI_COMM_WORLD, &status
  );
    
    MPI_Type_free( &section_row);
    //#########################################
    // Task-1(b)
    //########################################
    // TODO: define type MPI_Type_vector 
    MPI_Datatype section_col;
    MPI_Type_vector( my , 1 , mx , MPI_INT , &section_col);
    MPI_Type_commit( &section_col);

    int send_col = mx -1;
    int recv_col = 0;

    MPI_Sendrecv( 
      &m[0][send_col], 1, section_col, dest, 1,
      &m[0][recv_col], 1, section_col, src, 1,
      MPI_COMM_WORLD, &status
    );
    
    MPI_Type_free( &section_col);

    //Print After Process
    printf("Process %d Target array:\n", rank);
    for(j=0; j<my; j++) {
	    for(i=0; i<mx; i++) {
        printf("  %3d", m[j][i]);
	    }
      printf("\n");
    }
    printf("\n");
  
  MPI_Finalize();
  return 0;

}


