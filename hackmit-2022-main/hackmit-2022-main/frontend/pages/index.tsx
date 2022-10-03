import { Image } from "@chakra-ui/react";
import type { NextPage } from "next";
import { useRouter } from "next/router";
import { useEffect } from "react";

const Home: NextPage = () => {
  const router = useRouter();
  useEffect(() => {
    setTimeout(() => {
      router.push("/login");
    }, 1000);
  }, [router]);

  return (
    <div>
      <Image />
    </div>
  );
};

export default Home;
