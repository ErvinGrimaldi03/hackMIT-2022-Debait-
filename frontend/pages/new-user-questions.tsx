import {
  Button,
  Heading,
  IconButton,
  Image,
  Text,
  useToast,
  VStack,
} from "@chakra-ui/react";
import type { NextPage } from "next";
import { useRouter } from "next/router";
import { useEffect } from "react";
import { FcGoogle } from "react-icons/fc";
import { getAuth, GoogleAuthProvider, signInWithPopup } from "firebase/auth";

const Home: NextPage = () => {
	const questionsList = [
		"",
];
  return (
    <VStack my="80">
      <Heading>Welcome to Debaite</Heading>
      <Text>Login:</Text>
      <Button width="full" leftIcon={<FcGoogle />} onClick={onRegiser}>
        Register with Google
      </Button>
      <Button width="full" leftIcon={<FcGoogle />} onClick={onSignIn}>
        Login with Google
      </Button>
    </VStack>
  );
};

export default Home;
