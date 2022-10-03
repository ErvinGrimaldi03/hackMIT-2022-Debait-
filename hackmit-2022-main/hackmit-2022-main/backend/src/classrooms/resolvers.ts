import { update } from "lodash";
import { Arg, Authorized, Ctx, Mutation, Query, Resolver } from "type-graphql";
import {
  CreateOneClassroomResolver,
  StudentsOnClassrooms,
} from "../../prisma/generated/type-graphql";
import { AuthenticatedGraphQLContext } from "../auth/types";
import {
  FullClassroom,
  LimitedClassroom,
  TeacherClassroomUpdateInput,
} from "./types";

@Resolver()
class ClassroomsResolvers {
  @Authorized("teacher")
  @Mutation(() => FullClassroom, { nullable: true })
  async updateClassroom(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext,
    @Arg("classroomId") classroomId: string,
    @Arg("data")
    data: TeacherClassroomUpdateInput
  ): Promise<FullClassroom | null> {
    const { id: teacherId } = user;
    const existingClassroom = await prisma.classroom.findUnique({
      where: {
        id_teacherId: {
          teacherId,
          id: classroomId,
        },
      },
    });
    if (!existingClassroom.archived) {
      const updatedClassrooms = await prisma.classroom.update({
        where: {
          id_teacherId: {
            teacherId,
            id: classroomId,
          },
        },
        data,
      });
      return updatedClassrooms;
    } else {
      return null;
    }
  }

  @Authorized("teacher")
  @Mutation(() => FullClassroom, { nullable: true })
  async archiveClassroom(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext,
    @Arg("classroomId") classroomId: string
  ): Promise<FullClassroom | null> {
    const { id: teacherId } = user;
    return await prisma.classroom.update({
      where: {
        id_teacherId: {
          id: classroomId,
          teacherId,
        },
      },
      data: {
        archived: true,
      },
    });
  }

  @Authorized("teacher")
  @Mutation(() => FullClassroom, { nullable: true })
  async unarchiveClassroom(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext,
    @Arg("classroomId") classroomId: string
  ): Promise<FullClassroom | null> {
    const { id: teacherId } = user;
    return await prisma.classroom.update({
      where: {
        id_teacherId: {
          id: classroomId,
          teacherId,
        },
      },
      data: {
        archived: false,
      },
    });
  }

  @Authorized("teacher")
  @Mutation(() => StudentsOnClassrooms, { nullable: true })
  async addStudentToClassroom(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext,
    @Arg("classroomId") classroomId: string,
    @Arg("studentId") studentId: string
  ): Promise<StudentsOnClassrooms | null> {
    const { id: teacherId } = user;
    return await prisma.studentsOnClassrooms.create({
      data: {
        classroom: {
          connect: {
            id: classroomId,
          },
        },
        assignedBy: {
          connect: {
            id: teacherId,
          },
        },
        student: {
          connect: {
            id: studentId,
          },
        },
      },
    });
  }

  // To remove a student from a classroom, the teacher must be the one who assigned them
  @Authorized("teacher")
  @Mutation(() => StudentsOnClassrooms, { nullable: true })
  async removeStudentFromClassroom(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext,
    @Arg("classroomId") classroomId: string,
    @Arg("studentId") studentId: string
  ): Promise<StudentsOnClassrooms | null> {
    const { id: teahcerId } = user;
    return (
      await prisma.studentsOnClassrooms.deleteMany({
        where: {
          classroomId,
          studentId,
          assignedById: teahcerId,
        },
      })
    )[0];
  }

  @Authorized("teacher")
  @Query(() => FullClassroom, { nullable: true })
  async teacherClassroom(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext,
    @Arg("classroomId") classroomId: string
  ): Promise<FullClassroom | null> {
    const { id: teacherId } = user;
    const classroom = await prisma.classroom.findUnique({
      where: {
        id_teacherId: {
          teacherId,
          id: classroomId,
        },
      },
      include: {
        passes: true,
        studentToClassroomMappings: {
          select: {
            student: { include: { userProfile: true, passes: true } },
          },
        },
      },
    });

    if (classroom && !classroom.archived) {
      return {
        ...classroom,
        students: classroom.studentToClassroomMappings.map(
          (mapping) => mapping.student
        ),
        studentToClassroomMappings: null,
      };
    }
    return null;
  }

  @Authorized("teacher")
  @Query(() => [FullClassroom], { nullable: true })
  async teacherClassrooms(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext
  ): Promise<FullClassroom[] | null> {
    const { id: teacherId } = user;
    const classrooms = await prisma.classroom.findMany({
      where: { teacherId },
    });
    return classrooms;
  }

  @Authorized("student")
  @Query(() => LimitedClassroom, { nullable: true })
  async studentClassroom(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext,
    @Arg("classroomId", { nullable: true }) classroomId?: string,
    @Arg("classCode", { nullable: true }) classCode?: string
  ): Promise<LimitedClassroom | null> {
    if (classroomId || classCode) {
      const classroom = await prisma.classroom.findUnique({
        where: { id: classroomId, classCode: classCode },
      });
      if (!classroom.archived) {
        return classroom;
      }
    }
    return null;
  }

  @Authorized("student")
  @Query(() => [FullClassroom], { nullable: true })
  async studentClassrooms(
    @Ctx() { prisma, user }: AuthenticatedGraphQLContext
  ): Promise<FullClassroom[] | null> {
    const { id: studentId } = user;
    const classrooms = await prisma.classroom.findMany({
      where: {
        archived: false,
        studentToClassroomMappings: {
          some: {
            studentId,
          },
        },
      },
    });
    return classrooms;
  }
}

export const classroomsResolvers = [
  ClassroomsResolvers,
  CreateOneClassroomResolver,
];
