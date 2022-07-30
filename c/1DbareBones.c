/* Bare-bones 1D FDTD simulation with a hard source. */

#include <stdio.h>
#include <math.h>

#define SIZE 200

int main()
{
  double ez[SIZE] = {0.}, hy[SIZE] = {0.}, imp0 = 377.0;
  int qTime, maxTime = 250, mm;

  /* do time stepping */
  for (qTime = 0; qTime < maxTime; qTime++) {
    /* update magnetic field */
    for (mm = 0; mm < SIZE - 1; mm++)
      hy[mm] = hy[mm] + (ez[mm + 1] - ez[mm]) / imp0;

    /* update electric field */
    for (mm = 1; mm < SIZE; mm++)
      ez[mm] = ez[mm] + (hy[mm] - hy[mm - 1]) * imp0;

    /* hardwire a source node */
    ez[0] = exp(-(qTime - 30.) * (qTime - 30.) / 100.);

    printf("%g\n", ez[50]);
  } /* end of time-stepping */

  return 0;
}