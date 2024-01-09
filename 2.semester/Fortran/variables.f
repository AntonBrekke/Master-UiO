      module data
        implicit none 
        integer, parameter :: r4 = selected_real_kind(p=6, r=20)
        integer, parameter :: r8 = selected_real_kind(p=6, r=20)
        integer, parameter :: i16 = selected_int_kind(19)
    ! Need to end this program in order for it to run 