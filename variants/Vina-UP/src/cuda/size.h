#ifndef SIZE_H_
#define SIZE_H_


struct Small {
    static const int torsions_size = 32;
    static const int atoms_size = 128;
    static const int pairs_size = 2048;
    static const int atom_types_size = 32;
    static const int grids_size = 100 * 100 * 100;
    static const int r2_smooth_size = 15000;
};

#endif