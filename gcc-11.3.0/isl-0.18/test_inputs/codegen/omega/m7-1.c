for (int c0 = 1; c0 <= 9; c0 += 1) {
  if (c0 % 2 == 0) {
    for (int c1 = 1; c1 <= 9; c1 += 1) {
      s0(c1, c0);
      s1(c1, c0);
    }
  } else {
    for (int c1 = 1; c1 <= 9; c1 += 1)
      s0(c1, c0);
  }
}
