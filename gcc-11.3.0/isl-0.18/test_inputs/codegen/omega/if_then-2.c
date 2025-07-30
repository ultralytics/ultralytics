for (int c0 = 1; c0 <= 100; c0 += 1) {
  if (n >= 2) {
    s0(c0);
    for (int c1 = 1; c1 <= 100; c1 += 1) {
      s1(c0, c1);
      s2(c0, c1);
    }
  } else {
    for (int c1 = 1; c1 <= 100; c1 += 1)
      s2(c0, c1);
  }
}
