python3 get_spectrum_m_v2.py --flux_multiple  0.5 1 --R_0 0.5 1 --E_max 10 --m_max 30
python3 get_spectrum_m_v2.py --flux_multiple  2 --R_0 2 --E_max 10 --m_max 30
python3 get_spectrum_m_v2.py --flux_multiple  8 --R_0 4 --E_max 10 --m_max 30

python3 plot_spectrum.py --flux_multiple 0.5 1 --R_0 0.5 1 
python3 plot_spectrum.py --flux_multiple 2 --R_0 2
python3 plot_spectrum.py --flux_multiple 8 --R_0 4

python3 get_eigenstate_v2.py --flux 0.5 --R_0 0.5 -m 0 1 2 3 --r_max 12
python3 get_eigenstate_v2.py --flux 1 --R_0 1 -m 0 1 2 3 --r_max 12
python3 get_eigenstate_v2.py --flux 2 --R_0 2 -m 0 1 2 3 --r_max 12
python3 get_eigenstate_v2.py --flux 8 --R_0 4 -m 0 1 2 3 --r_max 12