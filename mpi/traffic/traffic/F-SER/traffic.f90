program traffic

  use trafficlib

  implicit none

  integer, parameter :: NCELL = 100000
  integer, parameter :: maxiter = 200000000/NCELL
  integer, parameter :: printfreq = maxiter/10

  integer :: i, iter, nmove, ncars
  real    :: density

  integer, allocatable, dimension(:) :: newroad, oldroad

  double precision :: tstart, tstop

! Set target density of cars

  density = 0.52

  write(*,*) 'Length of road is ', NCELL
  write(*,*) 'Number of iterations is ', maxiter
  write(*,*) 'Target density of cars is ', density

! Allocate arrays

  allocate(newroad(0:NCELL+1))
  allocate(oldroad(0:NCELL+1))

! Initialise road accordingly using random number generator

  write(*,*) 'Initialising ...'

  ncars = initroad(oldroad(1), NCELL, density, seed)

  write(*,*) '... done'

  write(*,*) 'Actual density of cars is ', float(ncars)/float(NCELL)
  write(*,*)

  tstart = gettime()

  do iter = 1, maxiter

     call updatebcs(oldroad, NCELL)

     nmove = updateroad(newroad, oldroad, NCELL)

! Copy new to old array

     do i = 1, NCELL

        oldroad(i) = newroad(i)

     end do

     if (mod(iter, printfreq) .eq. 0) then

        write(*,*) 'At iteration ', iter, ' average velocity is ', &
             float(nmove)/float(ncars)
     end if

  end do

  tstop = gettime()

  deallocate(newroad)
  deallocate(oldroad)

  write(*,*)
  write(*,*) 'Finished'
  write(*,*)
  write(*,*) 'Time taken was  ', tstop - tstart, ' seconds'
  write(*,*) 'Update rate was ', &
              1.d-6*float(NCELL)*float(maxiter)/(tstop-tstart), ' MCOPs'
  write(*,*)

end program traffic
