for (int c0 = 2; c0 <= 9; c0 += 1) {
  if (c0 >= 5) {
    s1(c0, 1);
    for (int c1 = 2; c1 <= 9; c1 += 1) {
      s1(c0, c1);
      s0(c0, c1);
    }
  } else {
    for (int c1 = 2; c1 <= 9; c1 += 1)
      s0(c0, c1);
  }
}
