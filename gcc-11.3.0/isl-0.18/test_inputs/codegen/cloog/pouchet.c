for (int c0 = 1; c0 <= floord(Ny, 2) + 2; c0 += 1)
  for (int c1 = max(c0 - 1, c0 / 2 + 1); c1 <= min(c0, (Ny + 2 * c0) / 4); c1 += 1) {
    if (Ny + 2 * c0 >= 4 * c1 + 1) {
      for (int c2 = 1; c2 <= 2; c2 += 1) {
        S1(c0 - c1, c1, 2 * c0 - 2 * c1, -2 * c0 + 4 * c1, c2);
        S2(c0 - c1, c1, 2 * c0 - 2 * c1, -2 * c0 + 4 * c1 - 1, c2);
      }
    } else {
      for (int c2 = 1; c2 <= 2; c2 += 1)
        S2((-Ny + 2 * c0) / 4, (Ny + 2 * c0) / 4, (-Ny / 2) + c0, Ny - 1, c2);
    }
  }
